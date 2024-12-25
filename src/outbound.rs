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

//! API for outbound message buffers.
//!
//! Real implementations of consensus protocols over unreliable
//! networks need to manage retransmission of protocol message in
//! order to be practical.  In order to be as efficient as possible,
//! this should utilize incoming protocol messages as acknowledgements
//! to the greatest degree possible.
//!
//! The [Outbound] trait represents objects that manage this
//! retransmission and inbound message processing.
use std::fmt::Display;
use std::hash::Hash;
use std::iter::FusedIterator;
use std::time::Instant;

use bitvec::order::Lsb0;
use bitvec::slice::IterOnes;
use bitvec::vec::BitVec;
use constellation_common::error::ScopedError;

use crate::parties::PartyIDMap;
use crate::round::RoundMsg;

/// Trait for outbound message buffers.
///
/// This is used to handle retransmission of messages over an
/// unreliable network.
pub trait Outbound<RoundID, Msg>
where
    RoundID: Clone + Display + Ord,
    Msg: RoundMsg<RoundID> {
    type PartyID: Clone + Display + From<usize> + Into<usize>;
    /// Type of errors that can result from
    /// [collect_outbound](Outbound::collect_outbound).
    type CollectOutboundError: Display + ScopedError;
    /// Type of errors that can result from [recv](Outbound::recv).
    type RecvError: Display;
    /// Type of configuration for creating instances.
    type Config: Clone;

    /// Create an instance from a number of parties and a
    /// configuration object.
    fn create(
        nparties: usize,
        config: Self::Config
    ) -> Self;

    /// Gather up outbound messages into `buf`.
    ///
    /// This will return the next point in time when there are
    /// expected to be messages available.
    fn collect_outbound<F>(
        &mut self,
        round: RoundID,
        func: F
    ) -> Result<Option<Instant>, Self::CollectOutboundError>
    where
        F: FnMut(OutboundGroup<Msg>);

    /// Process a message recieved from `party` and update internal state.
    fn recv(
        &mut self,
        msg: &Msg::Payload,
        party: &Self::PartyID
    ) -> Result<(), Self::RecvError>;

    /// Check whether this `Outbound` will ever send any more messages.
    fn finished(&self) -> bool;
}

/// An outgoing message, together with its recipients.
///
/// This structure represents a single protocol message to be sent to
/// one or more recipients of type `Party`.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OutboundGroup<Msg> {
    /// Bit vector of recipients to which to send.
    parties: BitVec,
    /// The message to send.
    msg: Msg
}

/// Iterator over recipients for an [OutboundGroup].
pub struct OutboundGroupPartiesIter<'a, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash {
    /// The [PartyIDMap] for the round from which this group originates.
    party_map: &'a PartyIDMap<PartyID, Party>,
    /// Iterator over the parties to which to send the message.
    parties: IterOnes<'a, usize, Lsb0>
}

impl<Msg> OutboundGroup<Msg> {
    #[inline]
    pub fn create(
        parties: BitVec,
        msg: Msg
    ) -> Self {
        OutboundGroup {
            parties: parties,
            msg: msg
        }
    }

    /// Get the message to be sent to the outbound group.
    #[inline]
    pub fn msg(&self) -> &Msg {
        &self.msg
    }

    /// Get an iterator over all the parties to which the message
    /// needs to be sent.
    #[inline]
    pub fn iter<'a, PartyID, Party>(
        &'a self,
        parties: &'a PartyIDMap<PartyID, Party>
    ) -> OutboundGroupPartiesIter<'a, PartyID, Party>
    where
        PartyID: Clone + From<usize> + Into<usize>,
        Party: Clone + Eq + Hash {
        OutboundGroupPartiesIter {
            parties: self.parties.iter_ones(),
            party_map: parties
        }
    }
}

impl<'a, PartyID, Party> Iterator
    for OutboundGroupPartiesIter<'a, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash
{
    type Item = &'a Party;

    #[inline]
    fn next(&mut self) -> Option<&'a Party> {
        match self.parties.next() {
            Some(idx) => self.party_map.idx_party(idx),
            None => None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.parties.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.parties.count()
    }

    #[inline]
    fn nth(
        &mut self,
        n: usize
    ) -> Option<&'a Party> {
        match self.parties.nth(n) {
            Some(idx) => self.party_map.idx_party(idx),
            None => None
        }
    }

    #[inline]
    fn last(self) -> Option<&'a Party> {
        match self.parties.last() {
            Some(idx) => self.party_map.idx_party(idx),
            None => None
        }
    }
}

impl<'a, PartyID, Party> ExactSizeIterator
    for OutboundGroupPartiesIter<'a, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash
{
    #[inline]
    fn len(&self) -> usize {
        self.parties.len()
    }
}

impl<'a, PartyID, Party> FusedIterator
    for OutboundGroupPartiesIter<'a, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash
{
}
