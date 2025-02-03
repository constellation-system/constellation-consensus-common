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

use serde::Deserialize;
use serde::Serialize;

#[derive(
    Clone, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename = "consensus-pool")]
#[serde(rename_all = "kebab-case")]
pub struct SingleRoundConfig<State> {
    #[serde(default)]
    backlog_size_hint: Option<usize>,
    #[serde(flatten)]
    state: State
}

impl<State> SingleRoundConfig<State> {
    #[inline]
    pub fn backlog_size_hint(&self) -> Option<usize> {
        self.backlog_size_hint
    }

    #[inline]
    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn take(self) -> (Option<usize>, State) {
        (self.backlog_size_hint, self.state)
    }
}
