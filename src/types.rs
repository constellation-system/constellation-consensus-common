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

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use constellation_common::codec::Decoder;
use constellation_common::codec::Encoder;
use constellation_common::codec::ISizeCodec;

use crate::round::RoundMsg;

pub trait RoundPartyIDTypes {
    /// Type of IDs used to index a [Party](RoundPartyIDTypes::Party) generally.
    ///
    /// This is used to index all parties that are known to the
    /// consensus protocol currently.  Within a given round, parties
    /// are given a [PartyRoundIdx](RoundPartyIDTypes::PartyRoundIdx)
    type PartyID: Clone + Display + Eq + Hash + From<usize> + Into<usize>;
    /// Type of IDs for individual rounds.
    type RoundID: Clone + Debug + Display + Hash + Eq + Ord;
}

pub trait RoundPartyIdxTypes: RoundPartyIDTypes {
    /// Type of IDs used to index a (RoundPartyIDTypes::Party) in a
    /// given round.
    ///
    /// The set of parties active in a given consensus round and the
    /// set of parties known to the protocol generally may differ over
    /// time.  This type exists so that the parties active in each
    /// round may be assigned a contiguous range of integers.
    type PartyRoundIdx: Clone + Display + From<usize> + Into<usize>;
}

pub trait RoundIDGenTypes: RoundPartyIDTypes {
    /// Generator used to obtain [RoundID](RoundPartyIDTypes::RoundID)s.
    type RoundIDs: Iterator<Item = Self::RoundID>;
}

pub trait ProtoMsgTypes: RoundPartyIDTypes {
    /// Type of operations decided on by the consensus protocol.
    type Oper;
    /// Type of protocol messages.
    type Msg: RoundMsg<Self::RoundID>;
}

pub trait PartyTypes: RoundPartyIdxTypes {
    /// Type of party data.
    ///
    /// This is in general a complex datatype used to identify
    /// parties.
    type Party: Clone + Display + Eq + Hash;
    /// Codec for encoding and decoding [Party](PartyTypes::Party)s.
    type PartyCodec: Decoder<Self::Party> + Encoder<Self::Party>;
 }

pub struct TestPartyTypes;

impl RoundPartyIDTypes for TestPartyTypes {
    type PartyID = usize;
    type RoundID = usize;
}

impl RoundPartyIdxTypes for TestPartyTypes {
    type PartyRoundIdx = usize;
}

impl PartyTypes for TestPartyTypes {
    type Party = isize;
    type PartyCodec = ISizeCodec;
}
