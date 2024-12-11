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

//! Common API for consensus protocols for the Constellation
//! distributed systems platform.
//!
//! This crate provides a base API for consensus protocol
//! implemenations that can be used by the Constellation consensus
//! component.
//!
//! To implement a new consensus protocol, see the [proto] module.
#![allow(clippy::redundant_field_names)]
#![allow(clippy::type_complexity)]

pub mod outbound;
pub mod parties;
pub mod proto;
pub mod round;
pub mod state;
