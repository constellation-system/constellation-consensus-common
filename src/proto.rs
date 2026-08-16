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

use constellation_common::config::CreateWithParam;
use constellation_common::error::ScopedError;

use crate::outbound::Outbound;
use crate::parties::PartiesMap;
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
use crate::state::ProtoStateRound;
use crate::state::ProtoStateSetParties;

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
    RoundID: Clone + Display + Ord {
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
    type RecvError: Display;
    type Out: Outbound<
            Types::RoundID,
            Self::Msg,
            PartyID = Types::PartyRoundIdx,
            RecvError = Self::RecvError,
            CollectOutboundError = Self::CollectOutboundError
        >;
}

/// Top-level trait for consensus protocol implementations.
pub trait ConsensusProto<Map, Types>:
    CreateWithParam<Types::PartyCodec>
where
    Types: PartyTypes + RoundPartyIdxTypes + RoundIDGenTypes,
    Map: PartiesMap<Types> {
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
        > + RoundsSetParties<Types>;
    /// Type of the single-round protocol state machine.
    type State: ProtoStateSetParties<Types>
        + ProtoStateRound<Types, Self::ProtoTypes>
        + ProtoState<Types>;
    /// Type of errors that can occur creating [Rounds].
    type RoundsError<PartiesErr>: Display
    where
        PartiesErr: Display;

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
    ) -> Result<Self::Rounds, Self::RoundsError<Map::RoundError>>;
}

/// Wrapper around [ConsensusProto] implementations for sharing
/// between threads.
///
/// The [ConsensusProtoRounds] implementation wraps the associated
/// [Rounds] implementation in [SharedRounds].
#[derive(Clone)]
pub struct SharedConsensusProto<Inner, Map, Types>
where
    Inner: ConsensusProto<Map, Types>,
    Types: PartyTypes + RoundPartyIdxTypes + RoundIDGenTypes,
    Map: PartiesMap<Types> {
    types: PhantomData<Types>,
    parties: PhantomData<Map>,
    inner: Inner
}

impl<Inner, Map, Types> CreateWithParam<Types::PartyCodec>
    for SharedConsensusProto<Inner, Map, Types>
where
    Inner: ConsensusProto<Map, Types>,
    Types: PartyTypes + RoundPartyIdxTypes + RoundIDGenTypes,
    Map: PartiesMap<Types>
{
    type Config = Inner::Config;
    type CreateError = Inner::CreateError;

    fn create(
        config: Self::Config,
        codec: Types::PartyCodec
    ) -> Result<Self, Self::CreateError> {
        let inner = Inner::create(config, codec)?;

        Ok(SharedConsensusProto {
            types: PhantomData,
            parties: PhantomData,
            inner: inner
        })
    }
}

impl<Inner, Map, Types> ConsensusProto<Map, Types>
    for SharedConsensusProto<Inner, Map, Types>
where
    Inner: ConsensusProto<Map, Types> + CreateWithParam<Types::PartyCodec>,
    Types: PartyTypes + RoundPartyIdxTypes + RoundIDGenTypes,
    Map: PartiesMap<Types>
{
    type ProtoTypes = Inner::ProtoTypes;
    type Rounds = SharedRounds<
        Inner::Rounds,
        Types,
        Inner::ProtoTypes,
        <Inner::State as ProtoState<Types>>::Oper
    >;
    type RoundsError<PartiesErr>
        = Inner::RoundsError<PartiesErr>
    where
        PartiesErr: Display;
    type State = Inner::State;

    fn rounds(
        &self,
        round_ids: Types::RoundIDs
    ) -> Result<Self::Rounds, Self::RoundsError<Map::RoundError>> {
        let rounds = self.inner.rounds(round_ids)?;

        Ok(SharedRounds::new(rounds))
    }
}
