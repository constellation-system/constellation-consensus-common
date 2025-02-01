// Copyright © 2024-25 The Johns Hopkins Applied Physics Laboratory LLC.
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
use std::hash::Hash;

use constellation_common::codec::Codec;

use crate::outbound::Outbound;
use crate::parties::Parties;
use crate::parties::PartyIDMap;
use crate::round::RoundMsg;

/// Trait for inter-round protocol states.
///
/// This trait allows protocol-specific state to be persisted between
/// rounds and to be updated with the results of a round.
pub trait ProtoState<RoundID, PartyID>: Sized {
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
        oper: Self::Oper
    ) -> Result<(), Self::UpdateError>
    where
        P: Parties<RoundID, PartyID>;
}

/// Subtrait of [ProtoState] allowing inter-round protocol states to
/// be created from a configuration object.
pub trait ProtoStateSetParties<PartyID, PartyData, C>
where
    C: Codec<PartyData>,
    PartyID: Clone + Display + Eq + Hash + Into<usize> {
    /// Type of errors that can occur creating a `ProtoState`.
    type SetPartiesError: Display;

    /// Create from a configuration.
    ///
    /// This returns a map from new IDs to the corresponding old IDs,
    /// or `None` if the new ID is freshly-created.
    fn set_parties(
        &mut self,
        codec: C,
        self_party: PartyData,
        party_data: &[PartyData]
    ) -> Result<Vec<Option<PartyID>>, Self::SetPartiesError>;
}

/// Subtrait of [ProtoState] allowing individual round states to be
/// created.
pub trait ProtoStateRound<RoundID, PartyID, Msg, Out>:
    ProtoState<RoundID, PartyID>
where
    PartyID: Clone + Eq + Hash + From<usize> + Into<usize>,
    RoundID: Clone + Display + Ord,
    Out: Outbound<RoundID, Msg>,
    Msg: RoundMsg<RoundID> {
    /// Type of round states.
    type Round: RoundState<
        RoundID,
        Out::PartyID,
        Self::Oper,
        Msg::Payload,
        Self::Info,
        Out
    >;
    /// Type of non-mutable round state.
    type Info;
    /// Errors that can occur when creating a round state.
    type CreateRoundError: Display;

    /// Create a new round state and outbound message buffer.
    fn create_round(
        &mut self,
        parties: &PartyIDMap<Out::PartyID, PartyID>
    ) -> Result<Option<(Self::Round, Self::Info, Out)>, Self::CreateRoundError>;
}

/// Per-round protocol state machine.
///
/// This provides the interface for the core protocol state-machine
/// for a single round.
pub trait RoundState<RoundID, Party, Oper, Msg, Info, Out>: Sized {
    /// Process a protocol message.
    fn recv(
        self,
        out: &mut Out,
        info: &Info,
        round: &RoundID,
        party: &Party,
        msg: Msg
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
