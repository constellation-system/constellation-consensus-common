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

use std::collections::HashSet;
use std::collections::VecDeque;
use std::fmt::Display;
use std::hash::Hash;

use constellation_common::hashid::HashAlgo;
use constellation_common::hashid::HashID;
use log::debug;
use log::error;
use log::trace;

pub enum OperBatchResult<H, T> {
    Hashes(Vec<H>),
    None(T)
}

/// Trait for consensus operations that possibly can be turned into a
/// batch.
pub trait OperBatch<H>: Sized
where
    H: HashAlgo {
    type BatchError: Display;

    /// Try to convert into a batch.
    ///
    /// If the operation cannot be turned into a batch, an "error" is
    /// returned with the original content.
    fn take_batch(
        self,
        hash: &H
    ) -> Result<OperBatchResult<H::HashID, Self>, Self::BatchError>;
}

pub struct OperBatches<H>
where
    H: Clone + Display + Eq + Hash + HashID {
    /// Maximum size of batches.
    max_batch_size: usize,
    /// Set of all valid hashes.
    pending: HashSet<H>,
    /// Queue for pending hashes.
    queue: VecDeque<H>
}

impl<H> OperBatches<H>
where
    H: Clone + Display + Eq + Hash + HashID
{
    #[inline]
    pub fn new(max_batch_size: usize) -> Self {
        OperBatches {
            max_batch_size: max_batch_size,
            pending: HashSet::new(),
            queue: VecDeque::new()
        }
    }

    #[inline]
    pub fn with_capacity(
        max_batch_size: usize,
        size: usize
    ) -> Self {
        OperBatches {
            max_batch_size: max_batch_size,
            pending: HashSet::with_capacity(size),
            queue: VecDeque::with_capacity(size)
        }
    }

    /// Insert a set of hashes into the structure.
    pub fn submit_hashes<I>(
        &mut self,
        hashes: I
    ) where
        I: Iterator<Item = H> {
        for hash in hashes {
            if self.pending.insert(hash.clone()) {
                debug!(target: "oper-batches",
                       "adding new hash {}",
                       hash);

                self.queue.push_back(hash)
            } else {
                trace!(target: "oper-batches",
                       "hash {} already known",
                       hash);
            }
        }
    }

    /// Generate a batch from the pending hashes, if possible.
    ///
    /// This only returns `None` if there are no hashes available.
    pub fn get_batch<Oper>(&self) -> Option<Oper>
    where
        Oper: From<Vec<H>> {
        let mut hashes: Option<Vec<H>> = None;

        debug!(target: "oper-batches",
               "getting hashes for new round");

        // Filter the queue by what's actually in the live hash set.
        for hash in self
            .queue
            .iter()
            .filter(|ent| self.pending.contains(ent))
            .take(self.max_batch_size)
        {
            trace!(target: "oper-batches",
                   "adding hash {} to batch",
                   hash);

            match &mut hashes {
                Some(hashes) => {
                    hashes.push(hash.clone());
                }
                None => {
                    let mut vec = Vec::with_capacity(self.max_batch_size);

                    vec.push(hash.clone());
                    hashes = Some(vec);
                }
            }
        }

        hashes.map(Oper::from)
    }

    /// Remove a set of hashes from the structure.
    pub fn clear_hashes<'a, I>(
        &mut self,
        hashes: I
    ) where
        I: Iterator<Item = &'a H>,
        H: 'a {
        trace!(target: "oper-batches",
               "clearing committed hashes from state");

        // Remove the hashes from the live set.
        for hash in hashes {
            trace!(target: "oper-batches",
                   "clearing hash {}",
                   hash);

            let _ = self.pending.remove(hash);
        }

        // Clear out the front of the queue.
        while self
            .queue
            .front()
            .is_some_and(|hash| !self.pending.contains(hash))
        {
            if let Some(hash) = self.queue.pop_front() {
                trace!(target: "oper-batches",
                       "popped hash {} from queue",
                       hash);
            } else {
                error!(target: "oper-batches",
                       "queue should not have been empty");
            }
        }
    }
}
