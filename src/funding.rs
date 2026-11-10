// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! What the tokenizer and the parser allocate, each allocation charged to a
//! [`TokenizerFunding`] before it is made and made fallibly: refused, nothing is allocated.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};
use core::fmt;

use crate::tokenizer::TokenizerFunding;

/// An allocation of `needed` bytes that was refused: by the funding it was charged to, or by
/// the allocator. Nothing was allocated for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Refused {
    pub(crate) needed: usize,
}

/// Funding that refuses nothing: allocations the allocator refuses still fail as refused rather
/// than abort.
pub(crate) struct Unfunded;

impl TokenizerFunding for Unfunded {
    fn charge(&self, _bytes: usize) -> bool {
        true
    }

    fn discharge(&self, _bytes: usize) {}
}

pub(crate) static UNFUNDED: Unfunded = Unfunded;

/// A buffer that is grown: room for its items is charged before it is allocated.
pub(crate) trait Buffer: Default {
    /// The bytes one item takes.
    const ITEM: usize;
    fn filled(&self) -> usize;
    fn room(&self) -> usize;
    /// Room for exactly `additional` more items than it holds; `false`, and nothing allocated,
    /// if the allocator refused it.
    fn grow_exact(&mut self, additional: usize) -> bool;
}

impl Buffer for String {
    const ITEM: usize = 1;

    fn filled(&self) -> usize {
        self.len()
    }

    fn room(&self) -> usize {
        self.capacity()
    }

    fn grow_exact(&mut self, additional: usize) -> bool {
        self.try_reserve_exact(additional).is_ok()
    }
}

impl<T> Buffer for Vec<T> {
    const ITEM: usize = core::mem::size_of::<T>();

    fn filled(&self) -> usize {
        self.len()
    }

    fn room(&self) -> usize {
        self.capacity()
    }

    fn grow_exact(&mut self, additional: usize) -> bool {
        self.try_reserve_exact(additional).is_ok()
    }
}

/// Counts the bytes a text is written in, allocating nothing.
pub(crate) struct Measure(pub(crate) usize);

impl fmt::Write for Measure {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0 += s.len();
        Ok(())
    }
}

/// The funding everything is made through.
#[derive(Clone, Copy)]
pub(crate) struct Funding<'f>(pub(crate) &'f dyn TokenizerFunding);

impl<'f> Funding<'f> {
    /// `bytes` charged before they are allocated.
    pub(crate) fn charge(self, bytes: usize) -> Result<(), Refused> {
        if bytes == 0 || self.0.charge(bytes) {
            Ok(())
        } else {
            Err(Refused { needed: bytes })
        }
    }

    /// `bytes` charged before, now freed.
    pub(crate) fn discharge(self, bytes: usize) {
        if bytes > 0 {
            self.0.discharge(bytes);
        }
    }

    /// Room for `additional` more items in `buffer`, the whole block it grows into charged before
    /// it is allocated (the block it grew out of is freed only once that is), and the block it
    /// grew out of given back once it is freed. Its room at least doubles as it grows.
    pub(crate) fn reserve<B: Buffer>(
        self,
        buffer: &mut B,
        additional: usize,
    ) -> Result<(), Refused> {
        let (filled, room) = (buffer.filled(), buffer.room());
        if room - filled >= additional {
            return Ok(());
        }
        let wanted = filled
            .checked_add(additional)
            .ok_or(Refused { needed: usize::MAX })?;
        let grown = wanted.max(room.saturating_mul(2)).max(8);
        self.grow_to(buffer, grown)
    }

    /// Room for exactly `additional` more items in `buffer`, charged as [`Funding::reserve`]
    /// charges it.
    pub(crate) fn reserve_exact<B: Buffer>(
        self,
        buffer: &mut B,
        additional: usize,
    ) -> Result<(), Refused> {
        let (filled, room) = (buffer.filled(), buffer.room());
        if room - filled >= additional {
            return Ok(());
        }
        let wanted = filled
            .checked_add(additional)
            .ok_or(Refused { needed: usize::MAX })?;
        self.grow_to(buffer, wanted)
    }

    fn grow_to<B: Buffer>(self, buffer: &mut B, grown: usize) -> Result<(), Refused> {
        let (filled, room) = (buffer.filled(), buffer.room());
        let bytes = grown
            .checked_mul(B::ITEM)
            .ok_or(Refused { needed: usize::MAX })?;
        self.charge(bytes)?;
        if !buffer.grow_exact(grown - filled) {
            self.discharge(bytes);
            return Err(Refused { needed: bytes });
        }
        self.discharge(room * B::ITEM);
        Ok(())
    }

    /// An empty text, allocating nothing until it grows.
    pub(crate) fn text(self) -> Held<'f, String> {
        Held {
            buffer: String::new(),
            funding: self,
        }
    }

    /// An empty list, allocating nothing until it grows.
    pub(crate) fn list<T>(self) -> Held<'f, Vec<T>> {
        Held {
            buffer: Vec::new(),
            funding: self,
        }
    }

    pub(crate) fn string(self, text: &str) -> Result<Held<'f, String>, Refused> {
        let mut made = self.text();
        made.push_str(text)?;
        Ok(made)
    }

    /// A copy of `text` in exactly its own bytes.
    pub(crate) fn copy(self, text: &str) -> Result<Held<'f, String>, Refused> {
        let mut made = self.text();
        made.reserve_exact(text.len())?;
        made.buffer.push_str(text);
        Ok(made)
    }

    pub(crate) fn chars(
        self,
        chars: impl IntoIterator<Item = char>,
    ) -> Result<Held<'f, String>, Refused> {
        let mut made = self.text();
        for ch in chars {
            made.push_char(ch)?;
        }
        Ok(made)
    }

    /// `args` written out, measured first so exactly its bytes are charged and allocated.
    pub(crate) fn format(self, args: fmt::Arguments<'_>) -> Result<Held<'f, String>, Refused> {
        let mut measure = Measure(0);
        fmt::write(&mut measure, args).map_err(|_| Refused { needed: 0 })?;
        let mut made = self.text();
        made.reserve_exact(measure.0)?;
        fmt::write(&mut made.buffer, args).map_err(|_| Refused { needed: measure.0 })?;
        Ok(made)
    }

    /// A list of exactly `items`.
    pub(crate) fn vec_of<T, const N: usize>(
        self,
        items: [T; N],
    ) -> Result<Held<'f, Vec<T>>, Refused> {
        let mut made = self.list();
        made.reserve_exact(N)?;
        made.buffer.extend(items);
        Ok(made)
    }

    /// `value` in a block of its own, charged before it is allocated. The charge goes with the
    /// box: whoever frees it gives back `size_of::<T>()`.
    pub(crate) fn boxed<T>(self, value: T) -> Result<Box<T>, Refused> {
        if core::mem::size_of::<T>() == 0 {
            // Nothing is allocated for it.
            return Ok(Box::new(value));
        }
        let slot = self.vec_of([value])?;
        // Reserved exactly, the room is one item: boxing the slice reallocates nothing.
        debug_assert_eq!(slot.buffer.capacity(), 1);
        let raw = Box::into_raw(slot.keep().into_boxed_slice()) as *mut T;
        // SAFETY: the boxed slice holds exactly one `T`, in a block allocated for room for one
        // `T`, whose layout `Layout::array::<T>(1)` is `Layout::new::<T>()`: the layout a
        // `Box<T>` frees it with. The slice's ownership moves into the box.
        Ok(unsafe { Box::from_raw(raw) })
    }
}

/// A buffer being grown, its room charged to its funding: given back as it is freed, unless it
/// is kept by what is made of it.
pub(crate) struct Held<'f, B: Buffer> {
    pub(crate) buffer: B,
    pub(crate) funding: Funding<'f>,
}

impl<B: Buffer> Held<'_, B> {
    pub(crate) fn reserve(&mut self, additional: usize) -> Result<(), Refused> {
        self.funding.reserve(&mut self.buffer, additional)
    }

    pub(crate) fn reserve_exact(&mut self, additional: usize) -> Result<(), Refused> {
        self.funding.reserve_exact(&mut self.buffer, additional)
    }

    /// The bytes its room was charged.
    pub(crate) fn charged(&self) -> usize {
        self.buffer.room() * B::ITEM
    }

    /// The buffer, its charge going with it: whoever frees it gives that back.
    pub(crate) fn keep(mut self) -> B {
        core::mem::take(&mut self.buffer)
    }
}

impl Held<'_, String> {
    pub(crate) fn push_char(&mut self, ch: char) -> Result<(), Refused> {
        self.reserve(ch.len_utf8())?;
        self.buffer.push(ch);
        Ok(())
    }

    pub(crate) fn push_str(&mut self, more: &str) -> Result<(), Refused> {
        self.reserve(more.len())?;
        self.buffer.push_str(more);
        Ok(())
    }

    /// Shortened to `len` bytes, keeping its room.
    pub(crate) fn truncate(&mut self, len: usize) {
        self.buffer.truncate(len);
    }
}

impl<T> Held<'_, Vec<T>> {
    pub(crate) fn push(&mut self, item: T) -> Result<(), Refused> {
        self.reserve(1)?;
        self.buffer.push(item);
        Ok(())
    }
}

impl<B: Buffer> core::ops::Deref for Held<'_, B> {
    type Target = B;

    fn deref(&self) -> &B {
        &self.buffer
    }
}

impl<B: Buffer> Drop for Held<'_, B> {
    fn drop(&mut self) {
        self.funding.discharge(self.charged());
    }
}
