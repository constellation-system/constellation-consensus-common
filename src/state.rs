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

//! Traits for protocol state machine.
//!
//! These traits provide both the inter-round and well as per-round
//! state for a consensus protocol.  The inter-round state manages
//! protocol-specific state that persists between rounds.  For
//! example, some protocols have "leaders" which are elected or
//! evicted by consensus round results; this would be maintained in
//! the inter-round state.  Inter-round state objects implement
//! [ProtoState] and [ProtoStateCreate].  Additionally, the
//! [ProtoStateRound] trait must be implemented to allow the
//! inter-round state to create a specific round.
//!
//! Per-round state refers to the protocol state-machine that manages
//! an individual round.  Implementations of per-round state machines
//! implement [RoundState].

use std::fmt::Display;
use std::time::Instant;

use constellation_common::error::ScopedError;

use crate::parties::Parties;
use crate::parties::PartyRoundIDMap;
use crate::parties::PartyTypes;
use crate::parties::RoundPartyIDTypes;
use crate::parties::RoundPartyIdxTypes;
use crate::proto::ConsensusProtoOutboundTypes;

/// Trait for inter-round protocol states.
///
/// This trait allows protocol-specific state to be persisted between
/// rounds and to be updated with the results of a round.
pub trait ProtoState<Types>: Sized
where
    Types: RoundPartyIDTypes {
    /// Configuration for creating states.
    type Config;
    /// Type of state-update operations.
    type Oper;
    /// Type of errors that can occur creating a `ProtoState`.
    type CreateError: Display;
    /// Errors that can occur applying updates.
    type UpdateError: Display;

    /// Create from a configuration.
    fn create(config: Self::Config) -> Result<Self, Self::CreateError>;

    /// Apply an operation to update the state.
    fn update<P>(
        &mut self,
        parties: &mut P,
        oper: &Self::Oper
    ) -> Result<(), Self::UpdateError>
    where
        P: Parties<Types>;
}

/// Trait for recording requests in the protocol state.
pub trait ProtoStateSubmit<Req> {
    type SubmitError: Display + ScopedError;

    /// Record requests into the protocol state.
    ///
    /// This will result in the local protocol state now knowing about
    /// these requests, and eventually forwarding them to other nodes.
    ///
    /// # Parameters
    ///
    /// - `reqs`: [Iterator] of requests to submit.
    fn submit_reqs<I>(
        &mut self,
        reqs: I
    ) -> Result<(), Self::SubmitError>
    where
        I: Iterator<Item = Req>;
}

/// Subtrait of [ProtoState] allowing inter-round protocol states to
/// be created from a configuration object.
pub trait ProtoStateSetParties<Types>
where
    Types: PartyTypes + RoundPartyIDTypes {
    /// Type of errors that can occur creating a `ProtoState`.
    type SetPartiesError: Display;

    /// Create from a configuration.
    ///
    /// This returns a map from new IDs to the corresponding old IDs,
    /// or `None` if the new ID is freshly-created.
    fn set_parties(
        &mut self,
        codec: Types::PartyCodec,
        self_party: Types::Party,
        party_data: &[Types::Party]
    ) -> Result<Vec<Option<Types::PartyID>>, Self::SetPartiesError>;
}

/// Subtrait of [ProtoState] allowing individual round states to be
/// created.
pub trait ProtoStateRound<Types, ProtoTypes>: ProtoState<Types>
where
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    Types: PartyTypes + RoundPartyIdxTypes {
    /// Type of round states.
    type Round: RoundStateRecv<Types, ProtoTypes, Self::Oper, Self::Info>
        + RoundStateNotify<ProtoTypes::Out, Self>;
    /// Type of non-mutable round state.
    type Info;
    /// Errors that can occur when creating a round state.
    type CreateRoundError: Display;

    /// Create a new round state and outbound message buffer.
    fn create_round(
        &mut self,
        parties: &PartyRoundIDMap<Types>
    ) -> Result<
        (Self::Round, Self::Info, ProtoTypes::Out, Option<Instant>),
        Self::CreateRoundError
    >;
}

/// Per-round protocol state machine.
///
/// This provides the interface for the core protocol state-machine
/// for a single round.
pub trait RoundState<Out>: Sized {
    fn time_update(
        self,
        out: &mut Out
    ) -> (Self, Option<Instant>);
}

pub trait RoundStateNotify<Out, State>: Sized {
    type NotifyError: Display + ScopedError;

    fn notify_update(
        self,
        state: &mut State,
        out: &mut Out
    ) -> Result<Self, Self::NotifyError>;
}

pub trait RoundStateRecv<Types, ProtoTypes, Oper, Info>:
    RoundState<ProtoTypes::Out>
where
    ProtoTypes: ConsensusProtoOutboundTypes<Types>,
    Types: PartyTypes + RoundPartyIdxTypes {
    /// Process a protocol message.
    fn recv(
        self,
        out: &mut ProtoTypes::Out,
        info: &Info,
        round: &Types::RoundID,
        party: &Types::PartyRoundIdx,
        msg: ProtoTypes::Payload
    ) -> RoundStateUpdate<Self, Oper>;
}

/// Trait for reporters for consensus round results.
///
/// This is used to report
pub trait RoundResultReporter<RoundID, Oper> {
    /// Errors that can occur reporting round results.
    type ReportError: Display;

    /// Report the outcome of a consensus round.
    fn report(
        &self,
        round: RoundID,
        oper: Oper
    ) -> Result<(), Self::ReportError>;
}

/// Outcome of processing a protocol message with
/// [recv](RoundState::recv).
pub enum RoundStateUpdate<Pending, Resolved> {
    /// The round is still unresolved.
    Pending {
        /// The new round state.
        pending: Pending
    },
    /// The round has been resolved.
    Resolved {
        /// The round result.
        resolved: Resolved
    }
}
