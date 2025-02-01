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
use std::fmt::Display;
use std::hash::Hash;
use std::marker::PhantomData;

use constellation_common::codec::Codec;

use crate::outbound::Outbound;
use crate::parties::PartiesMap;
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

/// Base trait for consensus protocol implementations.
///
/// This provides the means to create a consenus protocol
/// implementation.  Most of the meaningful definitions are found in
/// [ConsensusProtoRounds].
pub trait ConsensusProto<Party, PartyCodec>: Sized
where
    PartyCodec: Codec<Party> {
    /// Type of configuration objects used to create the protocol.
    type Config: Default;
    /// Type of errors that can occur creating a protocol instance.
    type CreateError: Display;

    /// Create an instance of the protocol from a configuration object.
    fn create(
        config: Self::Config,
        party_codec: PartyCodec
    ) -> Result<Self, Self::CreateError>;
}

/// Base trait for all consensus protocol implementations.
pub trait ConsensusProtoRounds<RoundIDs, PartyID, Party, PartyCodec, P>:
    ConsensusProto<Party, PartyCodec>
where
    RoundIDs: Iterator,
    RoundIDs::Item: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash + From<usize> + Into<usize>,
    Party: Clone + Display + Eq + Hash,
    P: PartiesMap<RoundIDs::Item, Self::RoundPartyIdx, PartyID>,
    PartyCodec: Codec<Party> {
    /// Type of protocol messages.
    type Msg: RoundMsg<RoundIDs::Item>;
    /// Type of outbound message structures.
    type Out: Outbound<RoundIDs::Item, Self::Msg>;
    /// Type of party indexes specific to a round.
    type RoundPartyIdx: Clone + Display + From<usize> + Into<usize>;
    /// Type of [Codec]s for consensus protocol messages.
    type Rounds: Rounds
        + RoundsAdvance<RoundIDs::Item>
        + RoundsUpdate<<Self::State as ProtoState<RoundIDs::Item, PartyID>>::Oper>
        + RoundsParties<
            RoundIDs::Item,
            PartyID,
            <Self::Out as Outbound<RoundIDs::Item, Self::Msg>>::PartyID
        > + RoundsRecv<
            RoundIDs::Item,
            PartyID,
            <Self::State as ProtoState<RoundIDs::Item, PartyID>>::Oper,
            Self::Msg
        > + RoundsSetParties<RoundIDs::Item, PartyID, Party, PartyCodec>;
    /// Protocol state machine.
    type State: ProtoStateSetParties<PartyID, Party, PartyCodec>
        + ProtoStateRound<RoundIDs::Item, PartyID, Self::Msg, Self::Out>
        + ProtoState<RoundIDs::Item, PartyID>;
    /// Type of errors that can occur creating [Rounds].
    type RoundsError<PartiesErr>: Display
    where
        PartiesErr: Display;

    /// Obtain the [Rounds] implementation for this consensus protocol.
    fn rounds(
        &self,
        round_ids: RoundIDs
    ) -> Result<Self::Rounds, Self::RoundsError<P::RoundError>>;
}

/// Wrapper around [ConsensusProto] implementations for sharing
/// between threads.
///
/// The [ConsensusProtoRounds] implementation wraps the associated
/// [Rounds] implementation in [SharedRounds].
#[derive(Clone)]
pub struct SharedConsensusProto<Inner, RoundIDs, PartyID, Party, PartyCodec, P>
where
    Inner: ConsensusProto<Party, PartyCodec>
        + ConsensusProtoRounds<RoundIDs, PartyID, Party, PartyCodec, P>,
    RoundIDs: Iterator,
    RoundIDs::Item: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash + From<usize> + Into<usize>,
    Party: Clone + Display + Eq + Hash,
    P: PartiesMap<RoundIDs::Item, Inner::RoundPartyIdx, PartyID>,
    PartyCodec: Codec<Party> {
    round_ids: PhantomData<RoundIDs>,
    party_id: PhantomData<PartyID>,
    party: PhantomData<Party>,
    party_codec: PhantomData<PartyCodec>,
    parties: PhantomData<P>,
    inner: Inner
}

impl<Inner, RoundIDs, PartyID, Party, PartyCodec, P>
    ConsensusProto<Party, PartyCodec>
    for SharedConsensusProto<Inner, RoundIDs, PartyID, Party, PartyCodec, P>
where
    Inner: ConsensusProto<Party, PartyCodec>
        + ConsensusProtoRounds<RoundIDs, PartyID, Party, PartyCodec, P>,
    RoundIDs: Iterator,
    RoundIDs::Item: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash + From<usize> + Into<usize>,
    Party: Clone + Display + Eq + Hash,
    P: PartiesMap<RoundIDs::Item, Inner::RoundPartyIdx, PartyID>,
    PartyCodec: Codec<Party>
{
    type Config = Inner::Config;
    type CreateError = Inner::CreateError;

    fn create(
        config: Self::Config,
        party_codec: PartyCodec
    ) -> Result<Self, Self::CreateError> {
        let inner = Inner::create(config, party_codec)?;

        Ok(SharedConsensusProto {
            round_ids: PhantomData,
            party_id: PhantomData,
            party: PhantomData,
            party_codec: PhantomData,
            parties: PhantomData,
            inner: inner
        })
    }
}

impl<Inner, RoundIDs, PartyID, Party, PartyCodec, P>
    ConsensusProtoRounds<RoundIDs, PartyID, Party, PartyCodec, P>
    for SharedConsensusProto<Inner, RoundIDs, PartyID, Party, PartyCodec, P>
where
    Inner: ConsensusProto<Party, PartyCodec>
        + ConsensusProtoRounds<RoundIDs, PartyID, Party, PartyCodec, P>,
    RoundIDs: Iterator,
    RoundIDs::Item: Clone + Display + Ord,
    PartyID: Clone + Display + Eq + Hash + From<usize> + Into<usize>,
    Party: Clone + Display + Eq + Hash,
    P: PartiesMap<RoundIDs::Item, Inner::RoundPartyIdx, PartyID>,
    PartyCodec: Codec<Party>
{
    type Msg = Inner::Msg;
    type Out = Inner::Out;
    type RoundPartyIdx = Inner::RoundPartyIdx;
    type Rounds = SharedRounds<
        Inner::Rounds,
        RoundIDs::Item,
        PartyID,
        <Inner::State as ProtoState<RoundIDs::Item, PartyID>>::Oper,
        Self::Msg,
        Self::Out
    >;
    type RoundsError<PartiesErr>
        = Inner::RoundsError<PartiesErr>
    where
        PartiesErr: Display;
    type State = Inner::State;

    fn rounds(
        &self,
        round_ids: RoundIDs
    ) -> Result<Self::Rounds, Self::RoundsError<P::RoundError>> {
        let rounds = self.inner.rounds(round_ids)?;

        Ok(SharedRounds::new(rounds))
    }
}
