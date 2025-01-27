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

//! Utilities for managing the parties in a consensus protocol.
//!
//! The handling of parties corresponding IDs is somewhat complex:
//!
//!  - Parties are typically described by fairly complex types such as strings,
//!    X509 Common Names, and other types.  These are too large to be used for
//!    most purposes.
//!
//!  - Parties are assigned dense IDs for the purposes of authentication,
//!    referencing, and adding/removing from the consensus pool.
//!
//!  - In general, the set of active parties in the consensus pool can vary from
//!    round to round.  Within an individual consensus round, the set of parties
//!    allowed to decide the round needs to be assigned a set of dense IDs that
//!    are *potentially different* from the more permanent set of IDs assigned
//!    to parties.
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt::Display;
use std::fmt::Error;
use std::fmt::Formatter;
use std::hash::Hash;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::replace;
use std::ops::Range;
use std::ops::RangeFrom;
use std::ops::RangeTo;

use constellation_common::error::ErrorScope;
use constellation_common::error::ScopedError;

/// Base trait for tracking parties through consensus rounds.
pub trait PartiesUpdate<Party> {
    /// Change the set of parties.
    ///
    /// The `parties` argument contains the new set of parties, and
    /// `map` contains an array that indicates which of the old
    /// parties corresponds to the new parties.  This is typically
    /// provided by
    /// [set_parties](crate::state::ProtoStateSetParties::set_parties).
    fn update_parties(
        &mut self,
        parties: Vec<Party>,
        map: &[Option<Party>]
    );
}

/// Trait for creating and advancing rounds when tracking parties.
pub trait PartiesRounds<Round> {
    /// Type of errors that can be returned by
    /// [advance_to](PartiesRounds::advance_to).
    type AdvanceError: Display;
    /// Type of errors that can be returned by functions that look up
    /// a specific round.
    type RoundError: Display;

    /// Create a round with id `RoundID`.
    fn next_round(
        &mut self,
        round_id: Round
    );

    /// Advance the lower bound for all parties to round `round`.
    ///
    /// This must be called in ascending order with regards to
    /// `round`.  Calling with a previous round will not necessarily
    /// restore the previous state.
    fn advance_to(
        &mut self,
        round: &Round
    ) -> Result<(), Self::AdvanceError>;

    /// Get a size hint for the number of parties in round `round`.
    ///
    /// This is an upper-bound, but is not guaranteed to be exact.
    fn nparties_hint(
        &self,
        round: &Round
    ) -> Result<usize, Self::RoundError>;
}

/// Trait for the set of parties participating in a consensus protocol.
///
/// This allows parties to be added and removed at given rounds.
pub trait Parties<Round, Party>:
    PartiesUpdate<Party> + PartiesRounds<Round> {
    /// Type of errors that can be returned by
    /// [start_party_at](DynamicParties::start_party_at) and
    /// [stop_party_at](Parties::stop_party_at).
    type PartyRoundError: Display;
    /// Type of iterators over parties.
    type PartiesIter<'a>: Iterator<Item = &'a Party>
    where
        Self: 'a,
        Party: 'a;

    /// Set `party` to be valid starting at round `round`.
    ///
    /// Calls to this and [stop_party_at](Parties::stop_party_at) must
    /// happen in ascending order, otherwise an error may occur.
    fn start_party_at(
        &mut self,
        party: Party,
        round: &Round
    ) -> Result<(), Self::PartyRoundError>;

    /// Set `party` to be invalid starting at round `round`.
    ///
    /// Calls to this and [start_party_at](DynamicParties::start_party_at) must
    /// happen in ascending order, otherwise an error may occur.
    fn stop_party_at(
        &mut self,
        party: Party,
        round: &Round
    ) -> Result<(), Self::PartyRoundError>;

    /// Get all active parties at round `round`.
    fn parties(
        &self,
        round: &Round
    ) -> Result<Self::PartiesIter<'_>, Self::RoundError>;
}

/// Trait for obtaining maps from parties to IDs for a given round.
///
/// In general, the dense IDs given to parties will vary from round to
/// round.  This means that a [PartyIDMap] needs to be obtained for
/// each round.
pub trait PartiesMap<Round, PartyID, Party>: Parties<Round, Party>
where
    PartyID: Clone + From<usize> + Into<usize> {
    /// Get the [PartyIDMap] for round `round`.
    #[inline]
    fn parties_map(
        &self,
        round: &Round
    ) -> Result<PartyIDMap<PartyID, Party>, Self::RoundError>
    where
        Party: Clone + Eq + Hash {
        Ok(PartyIDMap::from_iter(self.parties(round)?))
    }
}

/// A map from parties in a round to a dense integer range of `[1..n]`.
///
/// This is intended primarily for mapping party IDs to a more
/// convenient representation for per-round data structures.
pub struct PartyIDMap<PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash {
    /// Map from [Party] to index.
    fwd_map: HashMap<Party, PartyID>,
    /// Map from index to [Party].
    rev_map: Vec<Party>
}

/// Structure for intervals of rounds where a party is valid.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum RoundIntervals {
    /// Full range, valid everywhere.
    Full,
    /// A set of intervals of round numbers.
    Intervals {
        /// Starting interval, with no lower bound.
        start: Option<RangeTo<usize>>,
        /// Middle intervals.
        intervals: Vec<Range<usize>>,
        /// Ending interval, with no upper bound.
        end: Option<RangeFrom<usize>>
    }
}

/// Structure for tracking what parties are valid in which rounds.
///
/// This structure keeps track of the intervals for which a set of
/// parties are valid participants.  Parties can be added using
/// [start_party_at](Parties::start_party_at) and
/// [stop_party_at](Parties::stop_party_at), which must both be called in
/// ascending orer in the round for each party.  These will begin and
/// end valid ranges for a party.
///
/// The entire group can be shifted forward using
/// [advance_to](DynamicParties::advance_to) to discard information about
/// rounds that are concluded.
///
/// Finally, the set of valid parties for any round can be obtained with
/// [parties](DynamicParties::parties).
pub struct DynamicParties<Round, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash,
    Round: Clone + Eq + Hash {
    party_id: PhantomData<PartyID>,
    /// Map of rounds
    rounds: HashMap<Round, usize>,
    /// Hash table of parties.
    parties: HashMap<Party, RoundIntervals>,
    /// Next round.
    // ISSUE #1: replace this with shifting the RoundIntervals down.
    next: usize
}

/// Structure for representing a static set of parties.
///
/// This implements [Parties], but calls to
/// [start_party_at](Parties::start_party_at) and
/// [stop_party_at](Parties::stop_party_at) will always fail with an
/// error.
pub struct StaticParties<Party>
where
    Party: Clone + Eq + Hash {
    parties: Vec<Party>
}

/// Iterator for [DynamicParties].
pub struct DynamicPartiesIter<'a, Party>
where
    Party: Clone + Eq + Hash {
    iter: std::collections::hash_map::Iter<'a, Party, RoundIntervals>,
    round: usize
}

/// Errors that can occur trying to change the parties of a
/// [StaticParties].
#[derive(Clone, Debug)]
pub enum StaticPartiesError {
    /// Error indicating parties cannot be changed.
    Static
}

/// Errors that can occur in [DynamicParties].
#[derive(Clone, Debug)]
pub enum DynamicPartiesError {
    /// Round ID is not presently in the window.
    RoundOutsideWindow
}

impl<'a, Party> Iterator for DynamicPartiesIter<'a, Party>
where
    Party: Clone + Eq + Hash
{
    type Item = &'a Party;

    fn next(&mut self) -> Option<&'a Party> {
        match self.iter.next() {
            Some((party, rounds)) => {
                if rounds.contains(self.round) {
                    Some(party)
                } else {
                    self.next()
                }
            }
            None => None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }
}

impl<'a, Party> FusedIterator for DynamicPartiesIter<'a, Party> where
    Party: Clone + Eq + Hash
{
}

impl<PartyID, Party> PartyIDMap<PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash
{
    /// Create a `PartyIDMap` from an iterator over the parties in a
    /// round.
    #[inline]
    fn from_iter<'a, I>(iter: I) -> PartyIDMap<PartyID, Party>
    where
        I: Iterator<Item = &'a Party>,
        Party: 'a {
        Self::create(iter.cloned().collect())
    }

    /// Create a `PartyIDMap` from a [Vec] containing all the parties
    /// in a round.
    fn create(parties: Vec<Party>) -> PartyIDMap<PartyID, Party> {
        let mut map = HashMap::with_capacity(parties.len());

        for (i, party) in parties.iter().enumerate() {
            map.insert(party.clone(), i.into());
        }

        PartyIDMap {
            fwd_map: map,
            rev_map: parties
        }
    }

    /// Get the number of parties in the map.
    #[inline]
    pub fn nparties(&self) -> usize {
        self.rev_map.len()
    }

    /// Get the dense integer representing `party`.
    #[inline]
    pub fn party_idx(
        &self,
        party: &Party
    ) -> Option<&PartyID> {
        self.fwd_map.get(party)
    }

    /// Get the party represented by `idx`.
    #[inline]
    pub fn idx_party(
        &self,
        idx: usize
    ) -> Option<&Party> {
        if idx < self.rev_map.len() {
            Some(&self.rev_map[idx])
        } else {
            None
        }
    }
}

impl<Round, PartyID, Party> Default for DynamicParties<Round, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash,
    Round: Clone + Eq + Hash + Into<usize>
{
    #[inline]
    fn default() -> Self {
        DynamicParties {
            party_id: PhantomData,
            rounds: HashMap::new(),
            parties: HashMap::new(),
            next: 0
        }
    }
}

impl<Round, PartyID, Party> DynamicParties<Round, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash,
    Round: Clone + Eq + Hash + Into<usize>
{
    /// Create a new `DynamicParties`.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new `DynamicParties` with space reserved for `size` parties.
    #[inline]
    pub fn with_capacity(
        size: usize,
        nrounds: usize
    ) -> Self {
        DynamicParties {
            party_id: PhantomData,
            rounds: HashMap::with_capacity(nrounds),
            parties: HashMap::with_capacity(size),
            next: 0
        }
    }
}

impl<Party> Default for StaticParties<Party>
where
    Party: Clone + Eq + Hash
{
    #[inline]
    fn default() -> StaticParties<Party> {
        StaticParties {
            parties: Vec::new()
        }
    }
}

impl<Party> StaticParties<Party>
where
    Party: Clone + Eq + Hash
{
    #[inline]
    pub fn from_parties<'a, I>(iter: I) -> StaticParties<Party>
    where
        I: Iterator<Item = Party>,
        Party: 'a {
        StaticParties {
            parties: iter.collect()
        }
    }
}

impl<Party> PartiesUpdate<Party> for StaticParties<Party>
where
    Party: Clone + Eq + Hash
{
    #[inline]
    fn update_parties(
        &mut self,
        parties: Vec<Party>,
        _map: &[Option<Party>]
    ) {
        self.parties = parties
    }
}

impl<Round, Party> PartiesRounds<Round> for StaticParties<Party>
where
    Party: Clone + Eq + Hash
{
    type AdvanceError = Infallible;
    type RoundError = Infallible;

    #[inline]
    fn next_round(
        &mut self,
        _round_id: Round
    ) {
    }

    #[inline]
    fn advance_to(
        &mut self,
        _round: &Round
    ) -> Result<(), Infallible> {
        Ok(())
    }

    #[inline]
    fn nparties_hint(
        &self,
        _round: &Round
    ) -> Result<usize, Infallible> {
        Ok(self.parties.len())
    }
}

impl<Round, Party> Parties<Round, Party> for StaticParties<Party>
where
    Party: Clone + Eq + Hash
{
    type PartiesIter<'a> = std::slice::Iter<'a, Party>
    where Self: 'a,
          Party: 'a;
    type PartyRoundError = StaticPartiesError;

    #[inline]
    fn start_party_at(
        &mut self,
        _party: Party,
        _round: &Round
    ) -> Result<(), StaticPartiesError> {
        Err(StaticPartiesError::Static)
    }

    #[inline]
    fn stop_party_at(
        &mut self,
        _party: Party,
        _round: &Round
    ) -> Result<(), StaticPartiesError> {
        Err(StaticPartiesError::Static)
    }

    #[inline]
    fn parties(
        &self,
        _round: &Round
    ) -> Result<Self::PartiesIter<'_>, Infallible> {
        Ok(self.parties.iter())
    }
}

impl<Round, PartyID, Party> PartiesMap<Round, PartyID, Party>
    for StaticParties<Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash
{
}

impl<Round, PartyID, Party> PartiesUpdate<Party>
    for DynamicParties<Round, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash,
    Round: Clone + Eq + Hash
{
    fn update_parties(
        &mut self,
        parties: Vec<Party>,
        map: &[Option<Party>]
    ) {
        let new_parties = HashMap::with_capacity(parties.len());
        let mut old_parties = replace(&mut self.parties, new_parties);

        for (i, new_party) in parties.into_iter().enumerate() {
            if let Some(old_party) = &map[i] {
                if let Some(ent) = old_parties.remove(old_party) {
                    self.parties.insert(new_party, ent);
                }
            }
        }
    }
}

impl<Round, PartyID, Party> PartiesRounds<Round>
    for DynamicParties<Round, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash,
    Round: Clone + Eq + Hash
{
    type AdvanceError = DynamicPartiesError;
    type RoundError = DynamicPartiesError;

    #[inline]
    fn next_round(
        &mut self,
        round_id: Round
    ) {
        self.rounds.insert(round_id, self.next);
        self.next += 1;
    }

    fn advance_to(
        &mut self,
        round: &Round
    ) -> Result<(), DynamicPartiesError> {
        match self.rounds.get(round) {
            Some(round_idx) => {
                self.parties.retain(|_, rounds| {
                    rounds.advance_to(*round_idx);

                    !rounds.is_empty()
                });

                Ok(())
            }
            None => Err(DynamicPartiesError::RoundOutsideWindow)
        }
    }

    #[inline]
    fn nparties_hint(
        &self,
        _round: &Round
    ) -> Result<usize, DynamicPartiesError> {
        Ok(self.parties.len())
    }
}

impl<Round, PartyID, Party> Parties<Round, Party>
    for DynamicParties<Round, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash,
    Round: Clone + Eq + Hash
{
    type PartiesIter<'a> = DynamicPartiesIter<'a, Party>
    where Self: 'a,
          Party: 'a;
    type PartyRoundError = DynamicPartiesError;

    fn start_party_at(
        &mut self,
        party: Party,
        round: &Round
    ) -> Result<(), DynamicPartiesError> {
        match self.rounds.get(round) {
            Some(round_idx) => match self.parties.entry(party) {
                Entry::Occupied(mut ent) => ent
                    .get_mut()
                    .start_at(*round_idx)
                    .map_err(|_| DynamicPartiesError::RoundOutsideWindow),
                Entry::Vacant(ent) => {
                    ent.insert(RoundIntervals::starting_at(*round_idx));

                    Ok(())
                }
            },
            None => Err(DynamicPartiesError::RoundOutsideWindow)
        }
    }

    fn stop_party_at(
        &mut self,
        party: Party,
        round: &Round
    ) -> Result<(), DynamicPartiesError> {
        match self.rounds.get(round) {
            Some(round_idx) => match self.parties.entry(party) {
                Entry::Occupied(mut ent) => ent
                    .get_mut()
                    .stop_at(*round_idx)
                    .map_err(|_| DynamicPartiesError::RoundOutsideWindow),
                Entry::Vacant(_) => Err(DynamicPartiesError::RoundOutsideWindow)
            },
            None => Err(DynamicPartiesError::RoundOutsideWindow)
        }
    }

    fn parties(
        &self,
        round: &Round
    ) -> Result<Self::PartiesIter<'_>, DynamicPartiesError> {
        match self.rounds.get(round) {
            Some(round_id) => {
                let parties = DynamicPartiesIter {
                    iter: self.parties.iter(),
                    round: *round_id
                };

                Ok(parties)
            }
            None => Err(DynamicPartiesError::RoundOutsideWindow)
        }
    }
}

impl<Round, PartyID, Party> PartiesMap<Round, PartyID, Party>
    for DynamicParties<Round, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Eq + Hash,
    Round: Clone + Eq + Hash
{
}

impl RoundIntervals {
    #[cfg(test)]
    #[inline]
    fn empty() -> RoundIntervals {
        RoundIntervals::Intervals {
            start: None,
            end: None,
            intervals: vec![]
        }
    }

    #[cfg(test)]
    #[inline]
    fn full() -> RoundIntervals {
        RoundIntervals::Full
    }

    #[inline]
    fn starting_at(round: usize) -> RoundIntervals {
        RoundIntervals::Intervals {
            start: None,
            end: Some(round..),
            intervals: vec![]
        }
    }

    /// Check to see if this `RoundIntervals` contains this round.
    fn contains(
        &self,
        round: usize
    ) -> bool {
        match self {
            // Full interval, it's always contained.
            RoundIntervals::Full => true,
            // Check the starting interval.
            RoundIntervals::Intervals {
                start: Some(start), ..
            } if start.contains(&round) => true,
            // Check the ending interval.
            RoundIntervals::Intervals { end: Some(end), .. }
                if end.contains(&round) =>
            {
                true
            }
            // Try to find an interval that contains the round.
            RoundIntervals::Intervals { intervals, .. } => intervals
                .binary_search_by(|ent| {
                    if ent.contains(&round) {
                        Ordering::Equal
                    } else if ent.start > round {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                })
                .is_ok()
        }
    }

    /// Check if this `RoundIntervals` is empty.
    fn is_empty(&self) -> bool {
        match self {
            RoundIntervals::Intervals {
                start: None,
                end: None,
                intervals
            } => intervals.is_empty(),
            _ => false
        }
    }

    /// Stop being valid at a given round.
    ///
    /// This is only valid if `round` falls into the ending interval.
    /// If there is no ending interval (the round is already invalid),
    /// or it falls earlier, then an error will be returned indicating
    /// the most recent round at which this stopped being valid.
    fn stop_at(
        &mut self,
        round: usize
    ) -> Result<(), Option<usize>> {
        match self {
            // Full range; cut it off and make a starting interval.
            RoundIntervals::Full => {
                *self = RoundIntervals::Intervals {
                    start: Some(..round),
                    end: None,
                    intervals: vec![]
                };

                Ok(())
            }
            // Interval set; check if there is an ending interval.
            RoundIntervals::Intervals { end, intervals, .. } => match end {
                Some(end_ent) => {
                    intervals.push(end_ent.start..round);
                    *end = None;

                    Ok(())
                }
                // No ending interval.
                None => Err(intervals.last().map(|ent| ent.end))
            }
        }
    }

    /// Start being valid at a given round.
    ///
    /// This is only valid if `round` falls in a space greater than
    /// any defined interval.  If there is already an ending interval,
    /// this will return with an error.  If `round` falls into an
    /// interval that is already defined, an error indicating the end
    /// of the last interval will be returned.
    fn start_at(
        &mut self,
        round: usize
    ) -> Result<(), Option<usize>> {
        match self {
            RoundIntervals::Intervals { end, intervals, .. } => match end {
                None => match intervals.last() {
                    // The cutoff round is contained in the last interval.
                    Some(ent) if ent.end > round => Err(Some(ent.end)),
                    // Otherwise, we're good.
                    _ => {
                        *end = Some(round..);

                        Ok(())
                    }
                },
                // Ending interval exists.
                Some(_) => Err(None)
            },
            RoundIntervals::Full => Err(None)
        }
    }

    // ISSUE #2: this should shift down the intervals by `round`, to keep
    // everything in a [0, n] interval and avoid potential overflows.
    fn advance_to(
        &mut self,
        round: usize
    ) {
        match self {
            RoundIntervals::Intervals {
                start,
                intervals,
                end
            } if start.map_or(true, |start| start.end <= round) => {
                let len = intervals.len();

                for i in 0..len {
                    if intervals[i].end > round {
                        // This will be the last iteration either way.
                        if intervals[i].start <= round {
                            *start = Some(..intervals[i].end);

                            // ISSUE #3: This is inefficient, and should be
                            // replaced with a good data structure.
                            intervals.rotate_left(i + 1);
                            intervals.truncate(len - i - 1);

                            return;
                        } else {
                            // The new round falls wholly before the
                            // current interval.
                            *start = None;

                            // ISSUE #3: This is inefficient, and should be
                            // replaced with a good data structure.
                            intervals.rotate_left(i);
                            intervals.truncate(len - i);

                            return;
                        }
                    }
                }

                match end {
                    Some(end) if end.start <= round => {
                        *self = RoundIntervals::Full
                    }
                    _ => {
                        *start = None;
                        intervals.truncate(0);
                    }
                }
            }
            _ => {}
        }
    }
}

impl ScopedError for StaticPartiesError {
    #[inline]
    fn scope(&self) -> ErrorScope {
        match self {
            StaticPartiesError::Static => ErrorScope::Unrecoverable
        }
    }
}

impl Display for RoundIntervals {
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        match self {
            RoundIntervals::Intervals {
                start,
                intervals,
                end
            } => {
                let mut first = true;

                if let Some(start) = start {
                    write!(f, "..{}", start.end)?;
                    first = false
                };

                for ent in intervals.iter() {
                    if first {
                        write!(f, "{}..{}", ent.start, ent.end)?;
                    } else {
                        write!(f, ", {}..{}", ent.start, ent.end)?;
                    }
                    first = false;
                }

                if let Some(end) = end {
                    if first {
                        write!(f, "{}..", end.start)
                    } else {
                        write!(f, ", {}..", end.start)
                    }
                } else {
                    Ok(())
                }
            }
            RoundIntervals::Full => write!(f, "..")
        }
    }
}

impl<Round, PartyID, Party> Display for DynamicParties<Round, PartyID, Party>
where
    PartyID: Clone + From<usize> + Into<usize>,
    Party: Clone + Display + Eq + Hash,
    Round: Clone + Display + Eq + Hash + Into<usize>
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        for (party, rounds) in self.parties.iter() {
            write!(f, "party {}, valid in rounds: [{}]", party, rounds)?;
        }

        Ok(())
    }
}

impl Display for StaticPartiesError {
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        match self {
            StaticPartiesError::Static => write!(f, "parties cannot be changed")
        }
    }
}

impl Display for DynamicPartiesError {
    fn fmt(
        &self,
        f: &mut Formatter<'_>
    ) -> Result<(), Error> {
        match self {
            DynamicPartiesError::RoundOutsideWindow => {
                write!(f, "round is outside current window")
            }
        }
    }
}

#[cfg(test)]
use std::collections::HashSet;

#[test]
fn test_round_intervals_is_empty() {
    let intervals = RoundIntervals::empty();

    assert!(intervals.is_empty())
}

#[test]
fn test_round_intervals_full_not_is_empty() {
    let intervals = RoundIntervals::full();

    assert!(!intervals.is_empty())
}

#[test]
fn test_round_intervals_from_not_is_empty() {
    let intervals = RoundIntervals::Intervals {
        start: Some(..5),
        end: None,
        intervals: vec![]
    };

    assert!(!intervals.is_empty())
}

#[test]
fn test_round_intervals_to_not_is_empty() {
    let intervals = RoundIntervals::Intervals {
        start: None,
        end: Some(5..),
        intervals: vec![]
    };

    assert!(!intervals.is_empty())
}

#[test]
fn test_round_intervals_middle_not_is_empty() {
    let intervals = RoundIntervals::Intervals {
        start: None,
        end: None,
        intervals: vec![5..7]
    };

    assert!(!intervals.is_empty())
}

#[test]
fn test_round_intervals_contains_empty() {
    let intervals = RoundIntervals::empty();

    assert!(!intervals.contains(5))
}

#[test]
fn test_round_intervals_contains_full() {
    let intervals = RoundIntervals::full();

    assert!(intervals.contains(5))
}

#[test]
fn test_round_intervals_contains_start() {
    let intervals = RoundIntervals::Intervals {
        start: Some(..5),
        end: None,
        intervals: vec![]
    };

    assert!(intervals.contains(4))
}

#[test]
fn test_round_intervals_contains_end() {
    let intervals = RoundIntervals::Intervals {
        start: None,
        end: Some(5..),
        intervals: vec![]
    };

    assert!(intervals.contains(5))
}

#[test]
fn test_round_intervals_contains_simple_middle() {
    let intervals = RoundIntervals::Intervals {
        start: None,
        end: None,
        intervals: vec![4..7]
    };

    assert!(intervals.contains(5))
}

#[test]
fn test_round_intervals_contains_search_middle_low() {
    let intervals = RoundIntervals::Intervals {
        start: None,
        end: None,
        intervals: vec![2..6, 8..10, 12..15, 20..90]
    };

    assert!(intervals.contains(5))
}

#[test]
fn test_round_intervals_contains_search_middle_high() {
    let intervals = RoundIntervals::Intervals {
        start: None,
        end: None,
        intervals: vec![2..6, 8..10, 12..15, 20..90]
    };

    assert!(intervals.contains(20))
}

#[test]
fn test_round_intervals_contains_simple_middle_miss_low() {
    let intervals = RoundIntervals::Intervals {
        start: None,
        end: None,
        intervals: vec![4..7]
    };

    assert!(!intervals.contains(3))
}

#[test]
fn test_round_intervals_contains_simple_middle_miss_high() {
    let intervals = RoundIntervals::Intervals {
        start: None,
        end: None,
        intervals: vec![4..7]
    };

    assert!(!intervals.contains(8))
}

#[test]
fn test_round_intervals_contains_search_miss_middle_low() {
    let intervals = RoundIntervals::Intervals {
        start: None,
        end: None,
        intervals: vec![2..6, 8..10, 12..15, 20..90]
    };

    assert!(!intervals.contains(7))
}

#[test]
fn test_round_intervals_contains_search_miss_middle_high() {
    let intervals = RoundIntervals::Intervals {
        start: None,
        end: None,
        intervals: vec![2..6, 8..10, 12..15, 20..90]
    };

    assert!(!intervals.contains(15))
}

#[test]
fn test_round_intervals_stop_at_full() {
    let mut intervals = RoundIntervals::full();
    let expected = RoundIntervals::Intervals {
        start: Some(..5),
        end: None,
        intervals: vec![]
    };

    assert!(intervals.contains(6));

    intervals.stop_at(5).expect("Expected success");

    assert!(!intervals.contains(6));
    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_stop_at_empty() {
    let mut intervals = RoundIntervals::empty();

    assert!(intervals.stop_at(5).is_err())
}

#[test]
fn test_round_intervals_stop_at_intervals_end() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(7..),
        intervals: vec![]
    };
    let expected = RoundIntervals::Intervals {
        start: Some(..4),
        end: None,
        intervals: vec![7..9]
    };

    assert!(intervals.contains(10));

    intervals.stop_at(9).expect("Expected success");

    assert!(!intervals.contains(9));
    assert!(intervals.contains(8));
    assert!(intervals.contains(7));
    assert!(intervals.contains(3));
    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_stop_at_intervals_no_end() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: None,
        intervals: vec![7..10]
    };

    assert!(intervals.stop_at(9).is_err())
}

#[test]
fn test_round_intervals_start_at_full() {
    let mut intervals = RoundIntervals::full();

    assert!(intervals.start_at(5).is_err())
}

#[test]
fn test_round_intervals_start_at_empty() {
    let mut intervals = RoundIntervals::empty();
    let expected = RoundIntervals::Intervals {
        start: None,
        end: Some(5..),
        intervals: vec![]
    };

    assert!(!intervals.contains(6));

    intervals.start_at(5).expect("Expected success");

    assert!(intervals.contains(6));
    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_start_at_intervals_no_end() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: None,
        intervals: vec![]
    };
    let expected = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(9..),
        intervals: vec![]
    };

    assert!(!intervals.contains(10));

    intervals.start_at(9).expect("Expected success");

    assert!(!intervals.contains(8));
    assert!(intervals.contains(9));
    assert!(intervals.contains(10));
    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_start_at_intervals_end() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(12..),
        intervals: vec![7..10]
    };

    assert!(intervals.start_at(12).is_err())
}

#[test]
fn test_round_intervals_advance_to_full() {
    let mut intervals = RoundIntervals::full();
    let expected = RoundIntervals::Full;

    intervals.advance_to(5);

    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_advance_to_empty() {
    let mut intervals = RoundIntervals::empty();
    let expected = RoundIntervals::Intervals {
        start: None,
        end: None,
        intervals: vec![]
    };

    intervals.advance_to(5);

    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_advance_to_intervals_before_start() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: None,
        intervals: vec![6..9]
    };
    let expected = RoundIntervals::Intervals {
        start: Some(..4),
        end: None,
        intervals: vec![6..9]
    };

    intervals.advance_to(3);

    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_advance_to_intervals_after_start() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(25..),
        intervals: vec![6..9, 11..13, 16..19]
    };
    let expected = RoundIntervals::Intervals {
        start: None,
        end: Some(25..),
        intervals: vec![6..9, 11..13, 16..19]
    };

    intervals.advance_to(5);

    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_advance_to_intervals_in_first() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(25..),
        intervals: vec![6..9, 11..13, 16..19]
    };
    let expected = RoundIntervals::Intervals {
        start: Some(..9),
        end: Some(25..),
        intervals: vec![11..13, 16..19]
    };

    intervals.advance_to(6);

    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_advance_to_intervals_after_first() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(25..),
        intervals: vec![6..9, 11..13, 16..19]
    };
    let expected = RoundIntervals::Intervals {
        start: None,
        end: Some(25..),
        intervals: vec![11..13, 16..19]
    };

    intervals.advance_to(10);

    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_advance_to_intervals_in_second() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(25..),
        intervals: vec![6..9, 11..13, 16..19]
    };
    let expected = RoundIntervals::Intervals {
        start: Some(..13),
        end: Some(25..),
        intervals: vec![16..19]
    };

    intervals.advance_to(12);

    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_advance_to_intervals_after_second() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(25..),
        intervals: vec![6..9, 11..13, 16..19]
    };
    let expected = RoundIntervals::Intervals {
        start: None,
        end: Some(25..),
        intervals: vec![16..19]
    };

    intervals.advance_to(14);

    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_advance_to_intervals_in_third() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(25..),
        intervals: vec![6..9, 11..13, 16..19]
    };
    let expected = RoundIntervals::Intervals {
        start: Some(..19),
        end: Some(25..),
        intervals: vec![]
    };

    intervals.advance_to(18);

    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_advance_to_intervals_after_third() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(25..),
        intervals: vec![6..9, 11..13, 16..19]
    };
    let expected = RoundIntervals::Intervals {
        start: None,
        end: Some(25..),
        intervals: vec![]
    };

    intervals.advance_to(19);

    assert_eq!(intervals, expected);
}

#[test]
fn test_round_intervals_advance_to_intervals_in_end() {
    let mut intervals = RoundIntervals::Intervals {
        start: Some(..4),
        end: Some(25..),
        intervals: vec![6..9, 11..13, 16..19]
    };
    let expected = RoundIntervals::Full;

    intervals.advance_to(25);

    assert_eq!(intervals, expected);
}

#[test]
fn test_parties_start_at() {
    let mut parties: DynamicParties<usize, usize, isize> =
        DynamicParties::with_capacity(2, 2);
    let expected: HashSet<isize> = [-1, 0].iter().map(|x| *x).collect();

    parties.next_round(0);
    parties.next_round(1);
    parties.start_party_at(-1, &1).expect("Expected success");
    parties.start_party_at(0, &1).expect("Expected success");

    let actual: HashSet<isize> = parties
        .parties(&1)
        .expect("Expected success")
        .map(|x| *x)
        .collect();

    assert_eq!(expected, actual)
}

#[test]
fn test_parties_start_at_different() {
    let mut parties: DynamicParties<usize, usize, isize> =
        DynamicParties::with_capacity(2, 2);
    let expected: HashSet<isize> = [-1].iter().map(|x| *x).collect();

    parties.next_round(1);
    parties.next_round(2);
    parties.start_party_at(-1, &1).expect("Expected success");
    parties.start_party_at(0, &2).expect("Expected success");

    let actual: HashSet<isize> = parties
        .parties(&1)
        .expect("Expected success")
        .map(|x| *x)
        .collect();

    assert_eq!(expected, actual)
}

#[test]
fn test_parties_start_at_stop_at() {
    let mut parties: DynamicParties<usize, usize, isize> =
        DynamicParties::with_capacity(2, 5);
    let expected: HashSet<isize> = [].iter().map(|x| *x).collect();

    parties.next_round(0);
    parties.next_round(1);
    parties.next_round(2);
    parties.next_round(3);
    parties.next_round(4);
    parties.start_party_at(-1, &1).expect("Expected success");
    parties.start_party_at(0, &1).expect("Expected success");
    parties.stop_party_at(-1, &3).expect("Expected success");
    parties.stop_party_at(0, &3).expect("Expected success");

    let actual: HashSet<isize> = parties
        .parties(&4)
        .expect("Expected success")
        .map(|x| *x)
        .collect();

    assert_eq!(expected, actual)
}

#[test]
fn test_parties_start_at_stop_at_different() {
    let mut parties: DynamicParties<usize, usize, isize> =
        DynamicParties::with_capacity(2, 6);
    let expected: HashSet<isize> = [0].iter().map(|x| *x).collect();

    parties.next_round(0);
    parties.next_round(1);
    parties.next_round(2);
    parties.next_round(3);
    parties.next_round(4);
    parties.next_round(5);
    parties.start_party_at(-1, &1).expect("Expected success");
    parties.start_party_at(0, &1).expect("Expected success");
    parties.stop_party_at(-1, &3).expect("Expected success");
    parties.stop_party_at(0, &5).expect("Expected success");

    let actual: HashSet<isize> = parties
        .parties(&4)
        .expect("Expected success")
        .map(|x| *x)
        .collect();

    assert_eq!(expected, actual)
}

#[test]
fn test_parties_start_at_advance_to() {
    let mut parties: DynamicParties<usize, usize, isize> =
        DynamicParties::with_capacity(2, 2);
    let expected: HashSet<isize> = [-1, 0].iter().map(|x| *x).collect();

    parties.next_round(1);
    parties.next_round(2);
    parties.start_party_at(-1, &1).expect("Expected success");
    parties.start_party_at(0, &1).expect("Expected success");
    parties.advance_to(&2).expect("Expected success");

    let actual: HashSet<isize> = parties
        .parties(&1)
        .expect("Expected success")
        .map(|x| *x)
        .collect();

    assert_eq!(expected, actual)
}

#[test]
fn test_parties_start_at_different_advance_to() {
    let mut parties: DynamicParties<usize, usize, isize> =
        DynamicParties::with_capacity(2, 4);
    let expected: HashSet<isize> = [-1].iter().map(|x| *x).collect();

    parties.next_round(1);
    parties.next_round(2);
    parties.next_round(3);
    parties.next_round(4);
    parties.start_party_at(-1, &1).expect("Expected success");
    parties.start_party_at(0, &4).expect("Expected success");
    parties.advance_to(&2).expect("Expected success");

    let actual: HashSet<isize> = parties
        .parties(&3)
        .expect("Expected success")
        .map(|x| *x)
        .collect();

    assert_eq!(expected, actual)
}

#[test]
fn test_parties_start_at_stop_at_advance_to() {
    let mut parties: DynamicParties<usize, usize, isize> =
        DynamicParties::with_capacity(2, 5);
    let expected: HashSet<isize> = [].iter().map(|x| *x).collect();

    parties.next_round(1);
    parties.next_round(2);
    parties.next_round(3);
    parties.next_round(4);
    parties.next_round(5);
    parties.start_party_at(-1, &1).expect("Expected success");
    parties.start_party_at(0, &1).expect("Expected success");
    parties.stop_party_at(-1, &3).expect("Expected success");
    parties.stop_party_at(0, &3).expect("Expected success");
    parties.advance_to(&4).expect("Expected success");

    let actual: HashSet<isize> = parties
        .parties(&5)
        .expect("Expected success")
        .map(|x| *x)
        .collect();

    assert_eq!(expected, actual)
}

#[test]
fn test_parties_start_at_stop_at_different_advance_to() {
    let mut parties: DynamicParties<usize, usize, isize> =
        DynamicParties::with_capacity(2, 6);
    let expected: HashSet<isize> = [0].iter().map(|x| *x).collect();

    parties.next_round(1);
    parties.next_round(2);
    parties.next_round(3);
    parties.next_round(4);
    parties.next_round(5);
    parties.next_round(6);
    parties.start_party_at(-1, &1).expect("Expected success");
    parties.start_party_at(0, &1).expect("Expected success");
    parties.stop_party_at(-1, &3).expect("Expected success");
    parties.stop_party_at(0, &6).expect("Expected success");
    parties.advance_to(&4).expect("Expected success");

    let actual: HashSet<isize> = parties
        .parties(&5)
        .expect("Expected success")
        .map(|x| *x)
        .collect();

    assert_eq!(expected, actual)
}
