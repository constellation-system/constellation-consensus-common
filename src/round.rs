// Copyright © 2024-26 The Johns Hopkins Applied Physics Laboratory LLC.
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License,
// version 3, as published by the Free Software Foundation.  If you
// would like to purchase a commercial license for this software, please
// contact APL’s Tech Transfer at 240-592-0817 or
// techtransfer@jhuapl.edu.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public
// License along with this program.  If not, see
// <https://www.gnu.org/licenses/>.

//! Traits and types for managing consensus rounds.
//!
//! This is the top-level protocol state management API.  Consensus
//! protocols generally do *not* need to provide their own
//! implementation of this functionality.  They should use the
//! implementations provided here, and provide implementations of
//! helper objects.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::convert::Infallible;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Error;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use constellation_common::error::ErrorScope;
use constellation_common::error::ScopedError;
use constellation_common::error::WithMutexPoison;
use constellation_common::net::SharedMsgs;
use log::error;
use log::trace;

use crate::config::SingleRoundConfig;
use crate::outbound::Outbound;
use crate::outbound::OutboundGroup;
use crate::parties::PartiesMap;
use crate::parties::PartiesRounds;
use crate::parties::PartiesUpdate;
use crate::parties::PartyRoundIDMap;
use crate::parties::PartyTypes;
use crate::parties::RoundIDGenTypes;
use crate::parties::RoundPartyIDTypes;
use crate::parties::RoundPartyIdxTypes;
use crate::parties::StaticParties;
use crate::parties::StaticPartiesError;
use crate::proto::ConsensusProtoMsgTypes;
use crate::proto::ConsensusProtoOutboundTypes;
use crate::state::ProtoState;
use crate::state::ProtoStateRound;
use crate::state::ProtoStateSetParties;
use crate::state::ProtoStateSubmit;
use crate::state::RoundResultReporter;
use crate::state::RoundState;
use crate::state::RoundStateNotify;
use crate::state::RoundStateRecv;
use crate::state::RoundStateUpdate;

/// Trait for messages that have a round ID embedded.
///
/// # Type Parameters
///
/// - `RoundID`: Type of round IDs.
pub trait RoundMsg<RoundID>
where
    RoundID: Clone + Display + Ord {
    /// Type of message payloads.
    ///
    /// This should represent the main body of the protocol message.
    type Payload;

    /// Create an instance of the message.
    fn create(
        round: RoundID,
        payload: Self::Payload
    ) -> Self;

    /// Get the round ID.
    fn round_id(&self) -> RoundID;

    /// Get the message payload.
    fn payload(&self) -> &Self::Payload;

    /// Deconstruct the message into its round ID and payload.
    fn take(self) -> (RoundID, Self::Payload);
}

/// Base trait for objects that manage consensus rounds.
///
/// Most protocol implementations do *not* need to provide their own
/// implementations of this trait.
pub trait Rounds {
    /// Errors that can result from [recv](Rounds::recv).
    type TimeUpdateError: Display;
    // XXX Clear finished should also return an error.

    /// Clear out any rounds that have fully completed.
    ///
    /// This should drop any rounds that have been fully resolved.
    fn clear_finished(&mut self);

    /// Perform any state updates related to elapsed real time.
    fn time_update(&mut self)
    -> Result<Option<Instant>, Self::TimeUpdateError>;
}

pub trait RoundsSubmit<Req> {
    type SubmitError: Display + ScopedError;

    fn submit_reqs<I>(
        &mut self,
        reqs: I
    ) -> Result<(), Self::SubmitError>
    where
        I: Iterator<Item = Req>;
}

/// Subtrait of [Rounds] allowing advancement to the next round.
pub trait RoundsAdvance<RoundID>: Rounds
where
    RoundID: Clone + Display + Ord {
    /// Errors that can result from [advance](Rounds::advance).
    type AdvanceError: Display;

    /// Advance to the next round.
    fn advance(
        &mut self
    ) -> Result<Option<(RoundID, Option<Instant>)>, Self::AdvanceError>;
}

/// Subtrait of [Rounds] allowing an update to be applied to the round
/// state.
pub trait RoundsUpdate<Oper>: Rounds {
    /// Errors that can result from [update](Rounds::update).
    type UpdateError: Display;

    /// Update the inter-round state with `oper`.
    fn update(
        &mut self,
        oper: &Oper
    ) -> Result<(), Self::UpdateError>;
}

/// Trait for objects that manage the set of parties for a consensus
/// round.
///
/// Most protocol implementations do *not* need to provide their own
/// implementations of this trait.
///
/// # Type Parameters
///
/// - `Party`: Type of full party descriptions.
///
/// - `Codec`: Type of [Encoder]s and [Decoder]s for party data.
pub trait RoundsSetParties<Types>
where
    Types: PartyTypes {
    type SetPartiesError: Display;

    fn set_parties(
        &mut self,
        codec: Types::PartyCodec,
        self_party: Types::Party,
        party_data: &[Types::Party]
    ) -> Result<(), Self::SetPartiesError>;
}

/// Subtrait of [Rounds] for obtaining the ID mapping for a given
/// consensus round.
///
/// A set of "permanent" party IDs is maintained by the stream and
/// corresponding [Outbound] instance; however, the active parties for
/// a given round may vary over time, as parties are added or removed
/// from the pool.  Thus, it is necessary to maintain a mapping from
/// "permanent" party IDs to per-round party IDs.
pub trait RoundsParties<Types>: RoundsAdvance<Types::RoundID>
where
    Types: RoundPartyIdxTypes {
    /// Errors can occur getting active parties.
    type PartiesError: Display;

    /// Obtain a [PartyIDMap] for a given round.
    ///
    /// This maps permanent party IDs to per-round party IDs.
    fn round_parties(
        &self,
        round: &Types::RoundID
    ) -> Result<PartyRoundIDMap<Types>, Self::PartiesError>;
}

pub trait RoundsRecv<Types, ProtoTypes, Oper>:
    RoundsAdvance<Types::RoundID> + RoundsUpdate<Oper>
where
    Types: RoundPartyIDTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID> {
    /// Errors that can result from [recv](Rounds::recv).
    type RecvError<ReportError>: Display
    where
        ReportError: Display;

    /// Process an incoming protocol message from `party`.
    ///
    /// This will update both the outbound message buffer as well as
    /// the per-round state.
    fn recv<Reporter>(
        &mut self,
        reporter: &mut Reporter,
        party: &Types::PartyID,
        msg: ProtoTypes::Msg
    ) -> Result<(), Self::RecvError<Reporter::ReportError>>
    where
        Reporter: RoundResultReporter<Types::RoundID, Oper>;
}

/// Thread-safe wrapper around a [Rounds] implementation.
pub struct SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper> {
    types: PhantomData<Types>,
    proto_types: PhantomData<ProtoTypes>,
    oper: PhantomData<Oper>,
    inner: Arc<Mutex<Inner>>
}

struct SingleRoundCurr<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes> {
    round: Round<State::Round, Types, ProtoTypes, State::Oper, State::Info>,
    round_id: Types::RoundID
}

/// A [Rounds] instance that only tracks a single round.
///
/// This is intended for simple examples and testing.
pub struct SingleRound<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes> {
    state: State,
    round_ids: Types::RoundIDs,
    send_backlog: Vec<(Types::RoundID, ProtoTypes::Out)>,
    parties: StaticParties<Types::PartyID>,
    round: Option<SingleRoundCurr<State, Types, ProtoTypes>>
}

/// One round in a consensus protocol.
struct Round<State, Types, ProtoTypes, Oper, Info>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: RoundStateRecv<Types, ProtoTypes, Oper, Info> {
    oper: PhantomData<Oper>,
    /// The outbound messages for this round.
    outbound: ProtoTypes::Out,
    /// Non-mutable per-round state.
    info: Info,
    /// The protocol round state, if it's still alive.
    state: Option<State>
}

/// Errors that can occur recieving messages.
#[derive(Debug)]
pub enum RecvError<Recv, Report> {
    Recv { err: Recv },
    Report { err: Report }
}

/// Errors that can occur collecting outbound messages in [SingleRound].
#[derive(Debug)]
pub enum SingleRoundCollectOutboundError<RoundID, Inner> {
    Inner {
        err: Inner
    },
    Parties {
        err: SingleRoundPartiesError<RoundID>
    }
}

/// Errors that can occur receiving messages in [SingleRound].
#[derive(Debug)]
pub enum SingleRoundRecvError<RoundID, Inner, Party> {
    Inner {
        err: Inner
    },
    Parties {
        err: SingleRoundPartiesError<RoundID>
    },
    NotFound {
        party: Party
    }
}

/// Errors that can occur obtaining parties in [SingleRound].
#[derive(Debug)]
pub enum SingleRoundPartiesError<RoundID> {
    Parties { err: StaticPartiesError },
    BadRound { round: RoundID }
}

/// Errors that can occur creating a [SingleRound].
#[derive(Debug)]
pub enum SingleRoundCreateError<State, CreateRound> {
    CreateRound { err: CreateRound },
    Parties { err: StaticPartiesError },
    State { err: State },
    NoState,
    NoIDs
}

/// Errors that can occur advancing the round in [SingleRound].
#[derive(Debug)]
pub enum SingleRoundAdvanceError<CreateRound> {
    CreateRound { err: CreateRound },
    Parties { err: StaticPartiesError },
    NotFinished,
    NoIDs
}

#[derive(Debug)]
pub enum SingleRoundSubmitError<Submit, Notify> {
    Submit { err: Submit },
    Notify { err: Notify }
}

#[derive(Debug)]
pub enum SharedRoundsError<Inner> {
    Inner { err: Inner },
    MutexPoison
}

impl<Inner, Types, ProtoTypes, Oper>
    SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper>
{
    /// Create a `SharedRounds` from the inner [Rounds] instance.
    pub fn new(inner: Inner) -> Self {
        SharedRounds {
            types: PhantomData,
            proto_types: PhantomData,
            oper: PhantomData,
            inner: Arc::new(Mutex::new(inner))
        }
    }
}

impl<Inner, Types, ProtoTypes, Oper> Clone
    for SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper>
{
    #[inline]
    fn clone(&self) -> Self {
        SharedRounds {
            types: self.types,
            proto_types: self.proto_types,
            oper: self.oper,
            inner: self.inner.clone()
        }
    }
}

impl<Inner, Types, ProtoTypes, Oper, Req> RoundsSubmit<Req>
    for SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsSubmit<Req>
        + RoundsParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper>
{
    type SubmitError = WithMutexPoison<Inner::SubmitError>;

    fn submit_reqs<I>(
        &mut self,
        reqs: I
    ) -> Result<(), Self::SubmitError>
    where
        I: Iterator<Item = Req> {
        self.inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?
            .submit_reqs(reqs)
            .map_err(|err| WithMutexPoison::Inner { err: err })
    }
}

impl<Inner, Types, ProtoTypes, Oper> SharedMsgs<Types::PartyID, ProtoTypes::Msg>
    for SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper>
        + SharedMsgs<Types::PartyID, ProtoTypes::Msg>
{
    type MsgsError = WithMutexPoison<Inner::MsgsError>;

    fn msgs(
        &mut self,
        live: &HashSet<Types::PartyID>,
        now: Instant
    ) -> Result<
        (
            Option<Vec<(Vec<Types::PartyID>, Vec<ProtoTypes::Msg>)>>,
            Option<Instant>
        ),
        Self::MsgsError
    > {
        self.inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?
            .msgs(live, now)
            .map_err(|err| WithMutexPoison::Inner { err: err })
    }
}

impl<Inner, Types, ProtoTypes, Oper> Rounds
    for SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper>
{
    type TimeUpdateError = WithMutexPoison<Inner::TimeUpdateError>;

    fn clear_finished(&mut self) {
        match self.inner.lock() {
            Ok(mut guard) => guard.clear_finished(),
            Err(_) => {
                error!(target: "shared-rounds",
                       "mutex poisoned in clear_finished");
            }
        }
    }

    fn time_update(
        &mut self
    ) -> Result<Option<Instant>, Self::TimeUpdateError> {
        self.inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?
            .time_update()
            .map_err(|err| WithMutexPoison::Inner { err: err })
    }
}

impl<Inner, Types, ProtoTypes, Oper> RoundsAdvance<Types::RoundID>
    for SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper>
{
    type AdvanceError = WithMutexPoison<Inner::AdvanceError>;

    fn advance(
        &mut self
    ) -> Result<Option<(Types::RoundID, Option<Instant>)>, Self::AdvanceError>
    {
        self.inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?
            .advance()
            .map_err(|err| WithMutexPoison::Inner { err: err })
    }
}

impl<Inner, Types, ProtoTypes, Oper> RoundsUpdate<Oper>
    for SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper>
{
    type UpdateError = WithMutexPoison<Inner::UpdateError>;

    fn update(
        &mut self,
        oper: &Oper
    ) -> Result<(), Self::UpdateError> {
        self.inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?
            .update(oper)
            .map_err(|err| WithMutexPoison::Inner { err: err })
    }
}

impl<Inner, Types, ProtoTypes, Oper> RoundsSetParties<Types>
    for SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes + PartyTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsParties<Types>
        + RoundsSetParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper>
{
    type SetPartiesError = WithMutexPoison<Inner::SetPartiesError>;

    fn set_parties(
        &mut self,
        codec: Types::PartyCodec,
        self_party: Types::Party,
        party_data: &[Types::Party]
    ) -> Result<(), Self::SetPartiesError> {
        self.inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?
            .set_parties(codec, self_party, party_data)
            .map_err(|err| WithMutexPoison::Inner { err: err })
    }
}

impl<Inner, Types, ProtoTypes, Oper> RoundsParties<Types>
    for SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes + PartyTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper>
{
    type PartiesError = WithMutexPoison<Inner::PartiesError>;

    fn round_parties(
        &self,
        round: &Types::RoundID
    ) -> Result<PartyRoundIDMap<Types>, Self::PartiesError> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?;

        guard
            .round_parties(round)
            .map_err(|err| WithMutexPoison::Inner { err: err })
    }
}

impl<Inner, Types, ProtoTypes, Oper> RoundsRecv<Types, ProtoTypes, Oper>
    for SharedRounds<Inner, Types, ProtoTypes, Oper>
where
    Types: RoundPartyIdxTypes + PartyTypes,
    ProtoTypes: ConsensusProtoMsgTypes<Types::RoundID>,
    Inner: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<Oper>
        + RoundsParties<Types>
        + RoundsRecv<Types, ProtoTypes, Oper>
{
    type RecvError<ReportError>
        = WithMutexPoison<Inner::RecvError<ReportError>>
    where
        ReportError: Display;

    fn recv<Reporter>(
        &mut self,
        reporter: &mut Reporter,
        party: &Types::PartyID,
        msg: ProtoTypes::Msg
    ) -> Result<(), Self::RecvError<Reporter::ReportError>>
    where
        Reporter: RoundResultReporter<Types::RoundID, Oper> {
        self.inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?
            .recv(reporter, party, msg)
            .map_err(|err| WithMutexPoison::Inner { err: err })
    }
}

impl<State, Types, ProtoTypes, Oper, Info>
    Round<State, Types, ProtoTypes, Oper, Info>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: RoundStateRecv<Types, ProtoTypes, Oper, Info>
{
    #[inline]
    fn new(
        info: Info,
        state: State,
        outbound: ProtoTypes::Out
    ) -> Self {
        Round {
            oper: PhantomData,
            outbound: outbound,
            state: Some(state),
            info: info
        }
    }

    #[inline]
    fn collect_outbound<F>(
        &mut self,
        round: Types::RoundID,
        func: F
    ) -> Result<Option<Instant>, ProtoTypes::CollectOutboundError>
    where
        F: FnMut(OutboundGroup<ProtoTypes::Msg>) {
        self.outbound.collect_outbound(round, func)
    }

    #[inline]
    fn recv<Reporter>(
        &mut self,
        reporter: &mut Reporter,
        round: &Types::RoundID,
        party: &Types::PartyRoundIdx,
        msg: ProtoTypes::Payload
    ) -> Result<(), RecvError<ProtoTypes::RecvError, Reporter::ReportError>>
    where
        Reporter: RoundResultReporter<Types::RoundID, Oper> {
        // Log any acknowledgements in the incoming message.
        self.outbound
            .recv(&msg, party)
            .map_err(|err| RecvError::Recv { err: err })?;

        // Check if the round is still going.
        match self.state.take() {
            // The round is not yet resolved; apply the message.
            Some(state) => {
                match state.recv(
                    &mut self.outbound,
                    &self.info,
                    round,
                    party,
                    msg
                ) {
                    // Round is still going.
                    RoundStateUpdate::Pending { pending } => {
                        self.state = Some(pending);

                        Ok(())
                    }
                    // Round is resolved, though outbound messages may
                    // still be pending.
                    RoundStateUpdate::Resolved { resolved: oper } => reporter
                        .report(round.clone(), oper)
                        .map_err(|err| RecvError::Report { err: err })
                }
            }
            // The round is already resolved; nothing to do.
            None => {
                trace!(target: "consensus-round",
                       "discarding message from {} to concluded round",
                       party);

                Ok(())
            }
        }
    }

    /// Check whether this round can ever have more activity.
    #[inline]
    fn finished(&self) -> bool {
        self.state.is_none()
    }
}

impl<State, Types, ProtoTypes> SingleRound<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes>
{
    pub fn create(
        round_ids: Types::RoundIDs,
        round_config: SingleRoundConfig<State::Config>
    ) -> Result<
        Self,
        SingleRoundCreateError<State::CreateError, State::CreateRoundError>
    > {
        let (backlog_size, state_config) = round_config.take();
        // Create the initial protocol state.
        let proto_state = State::create(state_config)
            .map_err(|err| SingleRoundCreateError::State { err: err })?;
        let backlog = match backlog_size {
            Some(size) => Vec::with_capacity(size),
            None => Vec::new()
        };

        Ok(SingleRound {
            send_backlog: backlog,
            state: proto_state,
            parties: StaticParties::default(),
            round: None,
            round_ids: round_ids
        })
    }

    fn collect_outbound_msgs(
        group_map: &mut HashMap<Vec<Types::PartyID>, Vec<ProtoTypes::Msg>>,
        parties_map: &PartyRoundIDMap<Types>,
        group: OutboundGroup<ProtoTypes::Msg>
    ) {
        let mut party_idxs: Vec<Types::PartyID> =
            group.iter(parties_map).cloned().collect();

        party_idxs.sort();

        match group_map.entry(party_idxs) {
            Entry::Vacant(ent) => {
                ent.insert(vec![group.msg().clone()]);
            }
            Entry::Occupied(mut ent) => {
                ent.get_mut().push(group.msg().clone());
            }
        }
    }
}

impl<State, Types, ProtoTypes, Req> RoundsSubmit<Req>
    for SingleRound<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes> + ProtoStateSubmit<Req>
{
    type SubmitError = SingleRoundSubmitError<
        State::SubmitError,
        <State::Round as RoundStateNotify<ProtoTypes::Out, State>>::NotifyError
    >;

    #[inline]
    fn submit_reqs<I>(
        &mut self,
        reqs: I
    ) -> Result<(), Self::SubmitError>
    where
        I: Iterator<Item = Req> {
        self.state
            .submit_reqs(reqs)
            .map_err(|err| SingleRoundSubmitError::Submit { err: err })?;

        if let Some(curr) = &mut self.round {
            // If the round exists, update it.

            if let Some(state) = curr.round.state.take() {
                let new = state
                    .notify_update(&mut self.state, &mut curr.round.outbound)
                    .map_err(|err| SingleRoundSubmitError::Notify {
                        err: err
                    })?;

                curr.round.state = Some(new);
            }
        }

        Ok(())
    }
}

impl<State, Types, ProtoTypes> SharedMsgs<Types::PartyID, ProtoTypes::Msg>
    for SingleRound<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ProtoState<Types> + ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes> + ProtoStateSetParties<Types>
{
    type MsgsError = SingleRoundCollectOutboundError<
        Types::RoundID,
        ProtoTypes::CollectOutboundError
    >;

    // XXX Wire these parameters into the rest of the process
    fn msgs(
        &mut self,
        _live: &HashSet<Types::PartyID>,
        _now: Instant
    ) -> Result<
        (
            Option<Vec<(Vec<Types::PartyID>, Vec<ProtoTypes::Msg>)>>,
            Option<Instant>
        ),
        Self::MsgsError
    > {
        let mut group_map = HashMap::new();
        let mut min: Option<Instant> = None;
        let curr_parties = match &self.round {
            Some(curr) => {
                Some(self.round_parties(&curr.round_id).map_err(|err| {
                    SingleRoundCollectOutboundError::Parties { err: err }
                })?)
            }
            None => None
        };

        // Get the party may to convert the round-specific party IDs
        // back to parties.
        match (&mut self.round, curr_parties) {
            (Some(curr), Some(parties)) => {
                trace!(target: "single-round",
                       "collecting from current round {}",
                       curr.round_id);

                let curr_min = curr
                    .round
                    .collect_outbound(curr.round_id.clone(), |group| {
                        Self::collect_outbound_msgs(
                            &mut group_map,
                            &parties,
                            group
                        )
                    })
                    .map_err(|err| SingleRoundCollectOutboundError::Inner {
                        err: err
                    })?;

                min = match (min, curr_min) {
                    (Some(min), Some(curr_min)) => Some(min.min(curr_min)),
                    (Some(min), _) => Some(min),
                    (_, Some(curr_min)) => Some(curr_min),
                    _ => None
                };
            }
            (None, None) => {
                trace!(target: "single-round",
                       "no current round");
            }
            _ => {
                error!(target: "single-round",
                       "impossible mismatch between current round and parties");
            }
        };

        for i in 0..self.send_backlog.len() {
            let round = self.send_backlog[i].0.clone();
            let parties = self.round_parties(&round).map_err(|err| {
                SingleRoundCollectOutboundError::Parties { err: err }
            })?;
            let outbound = &mut self.send_backlog[i].1;

            trace!(target: "single-round",
                   "collecting from backlog round {}",
                   round);

            let curr = outbound
                .collect_outbound(round, |group| {
                    Self::collect_outbound_msgs(&mut group_map, &parties, group)
                })
                .map_err(|err| SingleRoundCollectOutboundError::Inner {
                    err: err
                })?;

            min = min.and_then(|min| curr.map(|curr| min.min(curr)))
        }

        let groups = if !group_map.is_empty() {
            Some(group_map.into_iter().collect())
        } else {
            None
        };

        Ok((groups, min))
    }
}

impl<State, Types, ProtoTypes> Rounds for SingleRound<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes>
{
    type TimeUpdateError = Infallible;

    fn clear_finished(&mut self) {
        self.send_backlog
            .retain(|(_, outbound)| !outbound.finished())
    }

    fn time_update(
        &mut self
    ) -> Result<Option<Instant>, Self::TimeUpdateError> {
        // Check if there is a current round.
        if let Some(curr) = &mut self.round {
            // If the round exists, update it.

            if let Some(state) = curr.round.state.take() {
                let (new, deadline) =
                    state.time_update(&mut curr.round.outbound);

                curr.round.state = Some(new);

                Ok(deadline)
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}

impl<State, Types, ProtoTypes> RoundsAdvance<Types::RoundID>
    for SingleRound<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes>
{
    type AdvanceError = SingleRoundAdvanceError<State::CreateRoundError>;

    fn advance(
        &mut self
    ) -> Result<
        Option<(Types::RoundID, Option<Instant>)>,
        SingleRoundAdvanceError<State::CreateRoundError>
    > {
        trace!(target: "single-round",
               "trying to advance round");

        if self.round.as_ref().is_none_or(|curr| curr.round.finished()) {
            // Get the next round ID.
            match self.round_ids.next() {
                Some(newid) => {
                    // Advance the parties structure to the next round.
                    self.parties.next_round(newid.clone());

                    // Create the next round state and outbound buffer.
                    let party_map = self
                        .parties
                        .parties_map(&newid)
                        .expect("infallible error");

                    let (round_state, info, outbound, deadline) =
                        self.state.create_round(&party_map).map_err(|err| {
                            SingleRoundAdvanceError::CreateRound { err: err }
                        })?;
                    let round = Round::new(info, round_state, outbound);
                    let curr = SingleRoundCurr {
                        round_id: newid.clone(),
                        round: round
                    };

                    if let Some(SingleRoundCurr { round, round_id }) =
                        self.round.replace(curr)
                    {
                        let outbound = round.outbound;

                        // Hang on to the old outbound if it's still
                        // going.
                        if !outbound.finished() {
                            trace!(target: "single-round",
                                   "retaining unfinished outbound buffer");

                            self.send_backlog.push((round_id, outbound));
                        }
                    }

                    trace!(target: "single-round",
                           "advanced to round {}",
                           newid);

                    Ok(Some((newid, deadline)))
                }
                None => {
                    // IDs are exhausted.
                    Err(SingleRoundAdvanceError::NoIDs)
                }
            }
        } else {
            Err(SingleRoundAdvanceError::NotFinished)
        }
    }
}

impl<State, Types, ProtoTypes> RoundsUpdate<State::Oper>
    for SingleRound<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes>
{
    type UpdateError = State::UpdateError;

    fn update(
        &mut self,
        oper: &State::Oper
    ) -> Result<(), Self::UpdateError> {
        self.state.update(&mut self.parties, oper)
    }
}

impl<State, Types, ProtoTypes> RoundsSetParties<Types>
    for SingleRound<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes> + ProtoStateSetParties<Types>
{
    type SetPartiesError = State::SetPartiesError;

    fn set_parties(
        &mut self,
        codec: Types::PartyCodec,
        self_party: Types::Party,
        party_data: &[Types::Party]
    ) -> Result<(), Self::SetPartiesError> {
        let remap = self.state.set_parties(codec, self_party, party_data)?;
        let parties = (0..party_data.len()).map(Types::PartyID::from).collect();

        self.parties.update_parties(parties, &remap);

        Ok(())
    }
}

impl<State, Types, ProtoTypes> RoundsParties<Types>
    for SingleRound<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes>
{
    type PartiesError = SingleRoundPartiesError<Types::RoundID>;

    fn round_parties(
        &self,
        round: &Types::RoundID
    ) -> Result<PartyRoundIDMap<Types>, Self::PartiesError> {
        Ok(self.parties.parties_map(round).expect("infallible error"))
    }
}

impl<State, Types, ProtoTypes> RoundsRecv<Types, ProtoTypes, State::Oper>
    for SingleRound<State, Types, ProtoTypes>
where
    Types: RoundPartyIdxTypes + PartyTypes + RoundIDGenTypes,
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    State: ProtoStateRound<Types, ProtoTypes>
{
    type RecvError<ReportError>
        = SingleRoundRecvError<
        Types::RoundID,
        RecvError<ProtoTypes::RecvError, ReportError>,
        Types::PartyID
    >
    where
        ReportError: Display;

    fn recv<Reporter>(
        &mut self,
        reporter: &mut Reporter,
        party: &Types::PartyID,
        msg: ProtoTypes::Msg
    ) -> Result<(), Self::RecvError<Reporter::ReportError>>
    where
        Reporter: RoundResultReporter<Types::RoundID, State::Oper> {
        // Get the round ID from the message.
        let (target_id, payload) = msg.take();
        // Get the party map to convert the party to the round-specific ID.
        let parties = self
            .round_parties(&target_id)
            .map_err(|err| SingleRoundRecvError::Parties { err: err })?;
        // Convert to the round-specific ID.
        let party = match parties.party_idx(party) {
            Some(idx) => Ok(idx),
            None => Err(SingleRoundRecvError::NotFound {
                party: party.clone()
            })
        }?;

        match &mut self.round {
            Some(SingleRoundCurr { round, round_id })
                if &target_id == round_id =>
            {
                trace!(target: "single-round",
                       "delivering to current round {}",
                       round_id);

                round
                    .recv(reporter, &target_id, party, payload)
                    .map_err(|err| SingleRoundRecvError::Inner { err: err })
            }
            _ => {
                trace!(target: "single-round",
                       "delivering to backlogged round {}",
                       target_id);

                // ISSUE #10: this is inefficient; do it some other way
                for (backlog_round, outbound) in self.send_backlog.iter_mut() {
                    if backlog_round == &target_id {
                        outbound.recv(&payload, party).map_err(|err| {
                            SingleRoundRecvError::Inner {
                                err: RecvError::Recv { err: err }
                            }
                        })?;
                    }
                }

                Ok(())
            }
        }
    }
}

impl<Submit, Notify> ScopedError for SingleRoundSubmitError<Submit, Notify>
where
    Submit: ScopedError,
    Notify: ScopedError
{
    fn scope(&self) -> ErrorScope {
        match self {
            SingleRoundSubmitError::Submit { err } => err.scope(),
            SingleRoundSubmitError::Notify { err } => err.scope()
        }
    }
}

impl<RoundID, Inner> ScopedError
    for SingleRoundCollectOutboundError<RoundID, Inner>
where
    Inner: ScopedError
{
    fn scope(&self) -> ErrorScope {
        match self {
            SingleRoundCollectOutboundError::Inner { err } => err.scope(),
            SingleRoundCollectOutboundError::Parties { err } => err.scope()
        }
    }
}

impl<RoundID> ScopedError for SingleRoundPartiesError<RoundID> {
    fn scope(&self) -> ErrorScope {
        match self {
            SingleRoundPartiesError::Parties { err } => err.scope(),
            SingleRoundPartiesError::BadRound { .. } => {
                ErrorScope::Unrecoverable
            }
        }
    }
}

impl<CreateRound> Display for SingleRoundAdvanceError<CreateRound>
where
    CreateRound: Display
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        match self {
            SingleRoundAdvanceError::CreateRound { err } => err.fmt(f),
            SingleRoundAdvanceError::Parties { err } => write!(f, "{}", err),
            SingleRoundAdvanceError::NotFinished => {
                write!(f, "round not finished")
            }
            SingleRoundAdvanceError::NoIDs => write!(f, "round IDs exhausted")
        }
    }
}

impl<State, CreateRound> Display for SingleRoundCreateError<State, CreateRound>
where
    CreateRound: Display,
    State: Display
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        match self {
            SingleRoundCreateError::CreateRound { err } => err.fmt(f),
            SingleRoundCreateError::Parties { err } => write!(f, "{}", err),
            SingleRoundCreateError::State { err } => err.fmt(f),
            SingleRoundCreateError::NoState => {
                write!(f, "no initial state created")
            }
            SingleRoundCreateError::NoIDs => write!(f, "round IDs exhausted")
        }
    }
}

impl<RoundID> Display for SingleRoundPartiesError<RoundID>
where
    RoundID: Display
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        match self {
            SingleRoundPartiesError::Parties { err } => write!(f, "{}", err),
            SingleRoundPartiesError::BadRound { round } => {
                write!(f, "wrong round {}", round)
            }
        }
    }
}

impl<Submit, Notify> Display for SingleRoundSubmitError<Submit, Notify>
where
    Submit: Display,
    Notify: Display
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        match self {
            SingleRoundSubmitError::Submit { err } => err.fmt(f),
            SingleRoundSubmitError::Notify { err } => err.fmt(f)
        }
    }
}

impl<RoundID, Inner> Display for SingleRoundCollectOutboundError<RoundID, Inner>
where
    RoundID: Display,
    Inner: Display
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        match self {
            SingleRoundCollectOutboundError::Parties { err } => err.fmt(f),
            SingleRoundCollectOutboundError::Inner { err } => err.fmt(f)
        }
    }
}

impl<RoundID, Inner, Parties> Display
    for SingleRoundRecvError<RoundID, Inner, Parties>
where
    RoundID: Display,
    Parties: Display,
    Inner: Display
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        match self {
            SingleRoundRecvError::Inner { err } => err.fmt(f),
            SingleRoundRecvError::Parties { err } => err.fmt(f),
            SingleRoundRecvError::NotFound { party } => {
                write!(f, "party {} not found", party)
            }
        }
    }
}

impl<Recv, Report> Display for RecvError<Recv, Report>
where
    Recv: Display,
    Report: Display
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        match self {
            RecvError::Recv { err } => err.fmt(f),
            RecvError::Report { err } => err.fmt(f)
        }
    }
}
