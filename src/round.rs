// Copyright © 2024 The Johns Hopkins Applied Physics Laboratory LLC.
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

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Display;
use std::fmt::Error;
use std::fmt::Formatter;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::replace;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use constellation_common::codec::DatagramCodec;
use constellation_common::error::ErrorScope;
use constellation_common::error::ScopedError;
use constellation_common::error::WithMutexPoison;
use constellation_common::net::SharedMsgs;
use log::error;
use log::trace;

use crate::outbound::Outbound;
use crate::outbound::OutboundGroup;
use crate::parties::Parties;
use crate::parties::PartiesMap;
use crate::parties::PartyIDMap;
use crate::parties::StaticParties;
use crate::parties::StaticPartiesError;
use crate::state::ProtoState;
use crate::state::ProtoStateCreate;
use crate::state::ProtoStateRound;
use crate::state::RoundResultReporter;
use crate::state::RoundState;
use crate::state::RoundStateUpdate;

/// Trait for messages that have a round ID embedded.
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

// ISSUE #7: split this trait into three parts, so each thread can
// have only the API it needs.

/// Trait for objects that manage consensus rounds.
///
/// Most protocol implementations do *not* need to provide their own
/// implementations of this trait.
pub trait Rounds<RoundID, PartyID, Oper, Msg, Out>
where
    RoundID: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID> {
    /// Errors can occur getting active parties.
    type PartiesError: Display;
    /// Errors that can result from [recv](Rounds::recv).
    type RecvError<ReportError>: Display
    where
        ReportError: Display;
    /// Errors that can result from [update](Rounds::update).
    type UpdateError: Display;
    /// Errors that can result from [advance](Rounds::advance).
    type AdvanceError: Display;

    /// Obtain a [PartyIDMap] for a given round.
    fn parties_map(
        &self,
        round: &RoundID
    ) -> Result<PartyIDMap<Out::PartyID, PartyID>, Self::PartiesError>;

    /// Process an incoming protocol message from `party`.
    ///
    /// This will update both the outbound message buffer as well as
    /// the per-round state.
    fn recv<Reporter>(
        &mut self,
        reporter: &mut Reporter,
        party: &PartyID,
        msg: Msg
    ) -> Result<(), Self::RecvError<Reporter::ReportError>>
    where
        Reporter: RoundResultReporter<RoundID, Oper>;

    /// Update the inter-round state with `oper`.
    fn update(
        &mut self,
        oper: Oper
    ) -> Result<(), Self::UpdateError>;

    /// Advance to the next round.
    fn advance(&mut self) -> Result<Option<RoundID>, Self::AdvanceError>;

    /// Clear out any rounds that have fully completed.
    ///
    /// This should drop any rounds that have been fully resolved.
    fn clear_finished(&mut self);
}

/// Thread-safe wrapper around a [Rounds] implementation.
pub struct SharedRounds<Inner, RoundID, PartyID, Oper, Msg, Out>
where
    Inner: Rounds<RoundID, PartyID, Oper, Msg, Out>,
    RoundID: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID> {
    round_id: PhantomData<RoundID>,
    party_id: PhantomData<PartyID>,
    msg: PhantomData<Msg>,
    out: PhantomData<Out>,
    oper: PhantomData<Oper>,
    inner: Arc<Mutex<Inner>>
}

/// A [Rounds] instance that only tracks a single round.
///
/// This is intended for simple examples and testing.
pub struct SingleRound<State, RoundIDs, PartyID, Msg, Out>
where
    State: ProtoStateRound<RoundIDs::Item, PartyID, Msg, Out>,
    RoundIDs: Iterator,
    RoundIDs::Item: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash + From<usize> + Into<usize>,
    Out: Outbound<RoundIDs::Item, Msg>,
    Msg: RoundMsg<RoundIDs::Item> {
    state: State,
    send_backlog: Vec<(RoundIDs::Item, Out)>,
    parties: StaticParties<PartyID>,
    round:
        Round<State::Round, RoundIDs::Item, State::Oper, Msg, State::Info, Out>,
    round_id: RoundIDs::Item,
    round_ids: RoundIDs
}

/// One round in a consensus protocol.
struct Round<State, RoundID, Oper, Msg, Info, Out>
where
    State: RoundState<RoundID, Out::PartyID, Oper, Msg::Payload, Info, Out>,
    RoundID: Clone + Display + Ord,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID> {
    round: PhantomData<RoundID>,
    oper: PhantomData<Oper>,
    msg: PhantomData<Msg>,
    /// The outbound messages for this round.
    outbound: Out,
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
pub enum SharedRoundsError<Inner> {
    Inner { err: Inner },
    MutexPoison
}

unsafe impl<Inner, RoundID, PartyID, Oper, Msg, Out> Send
    for SharedRounds<Inner, RoundID, PartyID, Oper, Msg, Out>
where
    Inner: Rounds<RoundID, PartyID, Oper, Msg, Out>,
    RoundID: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID>
{
}

unsafe impl<Inner, RoundID, PartyID, Oper, Msg, Out> Sync
    for SharedRounds<Inner, RoundID, PartyID, Oper, Msg, Out>
where
    Inner: Rounds<RoundID, PartyID, Oper, Msg, Out>,
    RoundID: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID>
{
}

impl<Inner, RoundID, PartyID, Oper, Msg, Out>
    SharedRounds<Inner, RoundID, PartyID, Oper, Msg, Out>
where
    Inner: Rounds<RoundID, PartyID, Oper, Msg, Out>,
    RoundID: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID>
{
    /// Create a `SharedRounds` from the inner [Rounds] instance.
    pub fn new(inner: Inner) -> Self {
        SharedRounds {
            round_id: PhantomData,
            party_id: PhantomData,
            msg: PhantomData,
            out: PhantomData,
            oper: PhantomData,
            inner: Arc::new(Mutex::new(inner))
        }
    }
}

impl<Inner, RoundID, PartyID, Oper, Msg, Out> Clone
    for SharedRounds<Inner, RoundID, PartyID, Oper, Msg, Out>
where
    Inner: Rounds<RoundID, PartyID, Oper, Msg, Out>,
    RoundID: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID>
{
    #[inline]
    fn clone(&self) -> Self {
        SharedRounds {
            round_id: self.round_id,
            party_id: self.party_id,
            msg: self.msg,
            out: self.out,
            oper: self.oper,
            inner: self.inner.clone()
        }
    }
}

impl<Inner, RoundID, PartyID, Oper, Msg, Out>
    SharedMsgs<PartyID, Msg>
    for SharedRounds<Inner, RoundID, PartyID, Oper, Msg, Out>
where
    Inner: Rounds<RoundID, PartyID, Oper, Msg, Out> +
           SharedMsgs<PartyID, Msg>,
    RoundID: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID>
{
    type MsgsError = WithMutexPoison<Inner::MsgsError>;

    fn msgs(
        &mut self
    ) -> Result<(Option<Vec<(Vec<PartyID>, Vec<Msg>)>>, Option<Instant>),
                Self::MsgsError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?;

        guard
            .msgs()
            .map_err(|err| WithMutexPoison::Inner { error: err })
    }
}

impl<Inner, RoundID, PartyID, Oper, Msg, Out>
    Rounds<RoundID, PartyID, Oper, Msg, Out>
    for SharedRounds<Inner, RoundID, PartyID, Oper, Msg, Out>
where
    Inner: Rounds<RoundID, PartyID, Oper, Msg, Out>,
    RoundID: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID>
{
    type AdvanceError = WithMutexPoison<Inner::AdvanceError>;
    type PartiesError = WithMutexPoison<Inner::PartiesError>;
    type RecvError<ReportError> = WithMutexPoison<Inner::RecvError<ReportError>>
    where ReportError: Display;
    type UpdateError = WithMutexPoison<Inner::UpdateError>;

    fn parties_map(
        &self,
        round: &RoundID
    ) -> Result<PartyIDMap<Out::PartyID, PartyID>, Self::PartiesError> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?;

        guard
            .parties_map(round)
            .map_err(|err| WithMutexPoison::Inner { error: err })
    }

    fn update(
        &mut self,
        oper: Oper
    ) -> Result<(), Self::UpdateError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?;

        guard
            .update(oper)
            .map_err(|err| WithMutexPoison::Inner { error: err })
    }

    fn advance(&mut self) -> Result<Option<RoundID>, Self::AdvanceError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?;

        guard
            .advance()
            .map_err(|err| WithMutexPoison::Inner { error: err })
    }

    fn recv<Reporter>(
        &mut self,
        reporter: &mut Reporter,
        party: &PartyID,
        msg: Msg
    ) -> Result<(), Self::RecvError<Reporter::ReportError>>
    where
        Reporter: RoundResultReporter<RoundID, Oper> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| WithMutexPoison::MutexPoison)?;

        guard
            .recv(reporter, party, msg)
            .map_err(|err| WithMutexPoison::Inner { error: err })
    }

    fn clear_finished(&mut self) {
        match self.inner.lock() {
            Ok(mut guard) => guard.clear_finished(),
            Err(_) => {
                error!(target: "shared-rounds",
                       "mutex poisoned in clear_finished");
            }
        }
    }
}

impl<State, RoundID, Oper, Msg, Info, Out>
    Round<State, RoundID, Oper, Msg, Info, Out>
where
    State: RoundState<RoundID, Out::PartyID, Oper, Msg::Payload, Info, Out>,
    RoundID: Clone + Display + Ord,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID>
{
    #[inline]
    fn new(
        info: Info,
        state: State,
        outbound: Out
    ) -> Self {
        Round {
            round: PhantomData,
            msg: PhantomData,
            oper: PhantomData,
            outbound: outbound,
            state: Some(state),
            info: info
        }
    }

    #[inline]
    fn collect_outbound<F>(
        &mut self,
        round: RoundID,
        func: F
    ) -> Result<Option<Instant>, Out::CollectOutboundError>
    where
        F: FnMut(OutboundGroup<Msg>) {
        self.outbound.collect_outbound(round, func)
    }

    #[inline]
    fn recv<Reporter>(
        &mut self,
        reporter: &mut Reporter,
        round: &RoundID,
        party: &Out::PartyID,
        msg: Msg::Payload
    ) -> Result<(), RecvError<Out::RecvError, Reporter::ReportError>>
    where
        Reporter: RoundResultReporter<RoundID, Oper> {
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

impl<State, RoundIDs, PartyID, Msg, Out>
    SingleRound<State, RoundIDs, PartyID, Msg, Out>
where
    State: ProtoState<RoundIDs::Item, PartyID>
        + ProtoStateRound<RoundIDs::Item, PartyID, Msg, Out>,
    RoundIDs: Iterator,
    RoundIDs::Item: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash + From<usize> + Into<usize> + Ord,
    Out: Outbound<RoundIDs::Item, Msg>,
    Msg: Clone + RoundMsg<RoundIDs::Item>
{
    pub fn create<Party, Codec>(
        mut round_ids: RoundIDs,
        codec: Codec,
        parties: StaticParties<PartyID>,
        self_party: Party,
        party_data: &[Party],
        state_config: State::Config
    ) -> Result<
        Self,
        SingleRoundCreateError<
            State::CreateError<StaticPartiesError>,
            State::CreateRoundError
        >
    >
    where
        State: ProtoStateCreate<RoundIDs::Item, PartyID, Party, Codec>,
        Party: Clone + Eq + Hash,
        Codec: DatagramCodec<Party> {
        match round_ids.next() {
            Some(round_id) => {
                // Create the initial protocol state.
                let mut proto_state = State::create(
                    state_config,
                    codec,
                    &round_id,
                    &parties,
                    self_party,
                    party_data
                )
                .map_err(|err| SingleRoundCreateError::State { err: err })?;
                // Create the first round.
                let party_map =
                    parties.parties_map(&round_id).map_err(|err| {
                        SingleRoundCreateError::Parties { err: err }
                    })?;
                let round =
                    proto_state.create_round(&party_map).map_err(|err| {
                        SingleRoundCreateError::CreateRound { err: err }
                    })?;
                let round = match round {
                    Some((round_state, info, outbound)) => {
                        Ok(Round::new(info, round_state, outbound))
                    }
                    None => Err(SingleRoundCreateError::NoState)
                }?;
                // ISSUE #9: take a size hint in the configuration.
                let backlog = Vec::new();

                Ok(SingleRound {
                    send_backlog: backlog,
                    state: proto_state,
                    parties: parties,
                    round: round,
                    round_id: round_id,
                    round_ids: round_ids
                })
            }
            None => Err(SingleRoundCreateError::NoIDs)
        }
    }

    fn collect_outbound_msgs(
        group_map: &mut HashMap<Vec<PartyID>, Vec<Msg>>,
        parties_map: &PartyIDMap<Out::PartyID, PartyID>,
        group: OutboundGroup<Msg>
    ) {
        let mut party_idxs: Vec<PartyID> =
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

impl<State, RoundIDs, PartyID, Msg, Out> SharedMsgs<PartyID, Msg>
    for SingleRound<State, RoundIDs, PartyID, Msg, Out>
where
    State: ProtoState<RoundIDs::Item, PartyID>
        + ProtoStateRound<RoundIDs::Item, PartyID, Msg, Out>,
    RoundIDs: Iterator,
    RoundIDs::Item: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash + From<usize> + Into<usize> + Ord,
    Out: Outbound<RoundIDs::Item, Msg>,
    Msg: Clone + RoundMsg<RoundIDs::Item>
{
    type MsgsError = SingleRoundCollectOutboundError<
        RoundIDs::Item,
        Out::CollectOutboundError
    >;

    fn msgs(
        &mut self
    ) -> Result<(Option<Vec<(Vec<PartyID>, Vec<Msg>)>>, Option<Instant>),
                Self::MsgsError> {
        let mut group_map = HashMap::new();

        // Get the party may to convert the round-specific party IDs
        // back to parties.
        let parties = self.parties_map(&self.round_id).map_err(|err| {
            SingleRoundCollectOutboundError::Parties { err: err }
        })?;

        trace!(target: "single-round",
               "collecting from current round {}",
               self.round_id);

        let mut min = self
            .round
            .collect_outbound(self.round_id.clone(), |group| {
                Self::collect_outbound_msgs(&mut group_map, &parties, group)
            })
            .map_err(|err| SingleRoundCollectOutboundError::Inner {
                err: err
            })?;

        for i in 0..self.send_backlog.len() {
            let round = self.send_backlog[i].0.clone();
            let parties = self.parties_map(&round).map_err(|err| {
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


impl<State, RoundIDs, PartyID, Msg, Out>
    Rounds<RoundIDs::Item, PartyID, State::Oper, Msg, Out>
    for SingleRound<State, RoundIDs, PartyID, Msg, Out>
where
    State: ProtoState<RoundIDs::Item, PartyID>
        + ProtoStateRound<RoundIDs::Item, PartyID, Msg, Out>,
    RoundIDs: Iterator,
    RoundIDs::Item: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash + From<usize> + Into<usize>,
    Out: Outbound<RoundIDs::Item, Msg>,
    Msg: RoundMsg<RoundIDs::Item>
{
    type AdvanceError = SingleRoundAdvanceError<State::CreateRoundError>;
    type PartiesError = SingleRoundPartiesError<RoundIDs::Item>;
    type RecvError<ReportError> = SingleRoundRecvError<
        RoundIDs::Item,
        RecvError<Out::RecvError, ReportError>,
        PartyID
    >
    where ReportError: Display;
    type UpdateError = State::UpdateError;

    fn parties_map(
        &self,
        round: &RoundIDs::Item
    ) -> Result<PartyIDMap<Out::PartyID, PartyID>, Self::PartiesError> {
        self.parties
            .parties_map(round)
            .map_err(|err| SingleRoundPartiesError::Parties { err: err })
    }

    fn update(
        &mut self,
        oper: State::Oper
    ) -> Result<(), Self::UpdateError> {
        self.state.update(&mut self.parties, oper)
    }

    fn advance(
        &mut self
    ) -> Result<
        Option<RoundIDs::Item>,
        SingleRoundAdvanceError<State::CreateRoundError>
    > {
        let round = &self.round;

        trace!(target: "single-round",
               "trying to advance round");

        if round.finished() {
            // Get the next round ID.
            match self.round_ids.next() {
                Some(newid) => {
                    // Advance the parties structure to the next round.
                    self.parties.next_round(newid.clone());

                    // Create the next round state and outbound buffer.
                    let party_map =
                        self.parties.parties_map(&round).map_err(|err| {
                            SingleRoundAdvanceError::Parties { err: err }
                        })?;
                    let round =
                        self.state.create_round(&party_map).map_err(|err| {
                            SingleRoundAdvanceError::CreateRound { err: err }
                        })?;

                    Ok(round.map(|(round_state, info, outbound)| {
                        let round = Round::new(info, round_state, outbound);
                        let round = replace(&mut self.round, round);
                        let outbound = round.outbound;
                        let oldid = replace(&mut self.round_id, newid);

                        // Hang on to the old outbound if it's still going.
                        if !outbound.finished() {
                            trace!(target: "single-round",
                                   "retaining unfinished outbound buffer");

                            self.send_backlog.push((oldid, outbound));
                        }

                        trace!(target: "single-round",
                               "advanced to round {}",
                               self.round_id);

                        self.round_id.clone()
                    }))
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

    fn recv<Reporter>(
        &mut self,
        reporter: &mut Reporter,
        party: &PartyID,
        msg: Msg
    ) -> Result<(), Self::RecvError<Reporter::ReportError>>
    where
        Reporter: RoundResultReporter<RoundIDs::Item, State::Oper> {
        // Get the round ID from the message.
        let (round, payload) = msg.take();
        // Get the party map to convert the party to the round-specific ID.
        let parties = self
            .parties_map(&round)
            .map_err(|err| SingleRoundRecvError::Parties { err: err })?;
        let party = match parties.party_idx(party) {
            Some(idx) => Ok(idx),
            None => Err(SingleRoundRecvError::NotFound {
                party: party.clone()
            })
        }?;

        if round == self.round_id {
            trace!(target: "single-round",
                   "delivering to current round {}",
                   self.round_id);

            self.round
                .recv(reporter, &round, party, payload)
                .map_err(|err| SingleRoundRecvError::Inner { err: err })
        } else {
            trace!(target: "single-round",
                   "delivering to backlogged round {}",
                   self.round_id);

            // ISSUE #10: this is inefficient; do it some other way
            for (backlog_round, outbound) in self.send_backlog.iter_mut() {
                if backlog_round == &round {
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

    fn clear_finished(&mut self) {
        self.send_backlog
            .retain(|(_, outbound)| !outbound.finished())
    }
}

impl<RoundID, Inner> ScopedError
    for SingleRoundCollectOutboundError<RoundID, Inner>
where Inner: ScopedError {

    fn scope(&self) -> ErrorScope {
        match self {
            SingleRoundCollectOutboundError::Inner { err } => err.scope(),
            SingleRoundCollectOutboundError::Parties { err } => err.scope(),
        }
    }
}

impl<RoundID> ScopedError for SingleRoundPartiesError<RoundID> {
    fn scope(&self) -> ErrorScope {
        match self {
            SingleRoundPartiesError::Parties { err } => err.scope(),
            SingleRoundPartiesError::BadRound { .. } =>
                ErrorScope::Unrecoverable
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
            SingleRoundAdvanceError::Parties { err } => err.fmt(f),
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
            SingleRoundCreateError::Parties { err } => err.fmt(f),
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
            SingleRoundPartiesError::Parties { err } => err.fmt(f),
            SingleRoundPartiesError::BadRound { round } => {
                write!(f, "wrong round {}", round)
            }
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
