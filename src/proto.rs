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

//! Top-level traits for implementing consensus protocols.
//!
//! Implementations of new consensus protocols should implement both
//! [ConsensusProto] and [ConsensusProtoRounds].  This involves
//! creating the following additional types:
//!
//!  - A configuration object, which will be used as [ConsensusProto::Config].
//!
//!  - A protocol message type, which will be used as
//!    [ConsensusProtoRounds::Msg].
//!
//!  - An outbound message buffer (see [Outbound]), which will be used as
//!    [ConsensusProtoRounds::Out].
//!
//!  - A protocol state machine (see [ProtoState]), which will be used as
//!    [ConsensusProtoRounds::State].
use std::fmt::Debug;
use std::fmt::Display;
use std::marker::PhantomData;

use constellation_common::config::Create;
use constellation_common::error::ScopedError;
use constellation_common::net::SharedMsgs;

use crate::outbound::Outbound;
use crate::parties::PartyTypes;
use crate::parties::RoundIDGenTypes;
use crate::parties::RoundPartyIdxTypes;
use crate::round::RoundMsg;
use crate::round::Rounds;
use crate::round::RoundsAdvance;
use crate::round::RoundsParties;
use crate::round::RoundsRecv;
use crate::round::RoundsSetParties;
use crate::round::RoundsUpdate;
use crate::round::SharedRounds;
use crate::state::ProtoState;
use crate::state::ProtoStateParties;
use crate::state::ProtoStateRound;

/// Type trait for consensus protocol messages.
///
/// This is one of the type traits that needs to be implemented as
/// part of a consensus protocol.
///
/// # Type Parameters
///
/// - `RoundID`: Type of round IDs.
pub trait ConsensusProtoMsgTypes<RoundID>
where
    RoundID: Clone + Debug + Display + Ord {
    type Payload;
    /// Type of protocol messages.
    type Msg: Clone + RoundMsg<RoundID, Payload = Self::Payload>;
}

/// Type trait for outbound message boxes for consensus protocols.
///
/// This is one of the type traits that needs to be implemented as
/// part of a consensus protocol.
///
/// # Type Parameters
///
/// - `Types`: [RoundPartyIdxTypes] type trait, describing round and party IDs.
pub trait ConsensusProtoOutboundTypes<Types>:
    ConsensusProtoMsgTypes<Types::RoundID>
where
    Types: RoundPartyIdxTypes {
    type CollectOutboundError: Debug + Display + ScopedError;
    type RecvError: Debug + Display;
    type Out: Outbound<
            Types::RoundID,
            Self::Msg,
            PartyID = Types::PartyRoundIdx,
            RecvError = Self::RecvError,
            CollectOutboundError = Self::CollectOutboundError
        >;
}

/// Top-level trait for consensus protocol implementations.
pub trait ConsensusProto<Types>: Create
where
    Types: PartyTypes + RoundPartyIdxTypes + RoundIDGenTypes {
    /// Type trait describing the protocol messages and outbound
    /// message box.
    type ProtoTypes: ConsensusProtoOutboundTypes<Types>;
    /// Data structure used to track protocol rounds.
    ///
    /// Implementors should generally use one of the already-existing
    /// implementations in [rounds](crate::rounds), such as
    /// [SingleRound](crate::rounds::SingleRound).
    type Rounds: Rounds
        + RoundsAdvance<Types::RoundID>
        + RoundsUpdate<<Self::State as ProtoState<Types>>::Oper>
        + RoundsParties<Types>
        + RoundsRecv<
            Types,
            Self::ProtoTypes,
            <Self::State as ProtoState<Types>>::Oper
        > + RoundsSetParties<Types>
        + SharedMsgs<
            Types::PartyID,
            <Self::ProtoTypes as ConsensusProtoMsgTypes<Types::RoundID>>::Msg
        >;
    /// Type of the single-round protocol state machine.
    type State: ProtoStateParties<Types>
        + ProtoStateRound<Types, Self::ProtoTypes>
        + ProtoState<Types>;
    /// Type of errors that can occur creating [Rounds].
    type RoundsError: Debug + Display;

    /// Obtain a protocol engine for this protocol.
    ///
    /// This is used by upstream users to obtain a protocol engine
    /// instance.
    ///
    /// # Parameters
    ///
    /// - `round_ids`: The round ID generator.
    fn rounds(
        &self,
        round_ids: Types::RoundIDs
    ) -> Result<Self::Rounds, Self::RoundsError>;
}

/// Wrapper around [ConsensusProto] implementations for sharing
/// between threads.
///
/// The [ConsensusProtoRounds] implementation wraps the associated
/// [Rounds] implementation in [SharedRounds].
#[derive(Clone)]
pub struct SharedConsensusProto<Inner, Types>
where
    Inner: ConsensusProto<Types>,
    Types: PartyTypes + RoundPartyIdxTypes + RoundIDGenTypes {
    types: PhantomData<Types>,
    inner: Inner
}

impl<Inner, Types> Create for SharedConsensusProto<Inner, Types>
where
    Inner: ConsensusProto<Types>,
    Types: PartyTypes + RoundPartyIdxTypes + RoundIDGenTypes
{
    type Config = Inner::Config;
    type CreateError = Inner::CreateError;

    fn create(config: Self::Config) -> Result<Self, Self::CreateError> {
        let inner = Inner::create(config)?;

        Ok(SharedConsensusProto {
            types: PhantomData,
            inner: inner
        })
    }
}

impl<Inner, Types> ConsensusProto<Types> for SharedConsensusProto<Inner, Types>
where
    Inner: ConsensusProto<Types> + Create,
    Types: PartyTypes + RoundPartyIdxTypes + RoundIDGenTypes
{
    type ProtoTypes = Inner::ProtoTypes;
    type Rounds = SharedRounds<
        Inner::Rounds,
        Types,
        Inner::ProtoTypes,
        <Inner::State as ProtoState<Types>>::Oper
    >;
    type RoundsError = Inner::RoundsError;
    type State = Inner::State;

    fn rounds(
        &self,
        round_ids: Types::RoundIDs
    ) -> Result<Self::Rounds, Self::RoundsError> {
        let rounds = self.inner.rounds(round_ids)?;

        Ok(SharedRounds::new(rounds))
    }
}
