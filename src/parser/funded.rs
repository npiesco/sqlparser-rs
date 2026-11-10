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

//! What the parser allocates: each allocation charged to the parser's funding before it is
//! made, and made fallibly, so that a refusal is a [`ParserError::OutOfMemory`] with nothing
//! allocated for it.
//!
//! The AST is plain `Box`, `Vec` and `String`, so no node can carry its own charge. The parser
//! counts instead what it charged for everything it made and handed out ([`Parser::kept`]):
//! the statements it returns (or the error it fails with) hold exactly that, to be given back
//! by whoever frees them. What a failed alternative made is freed with it, so it is given back
//! when the parser backtracks ([`Parser::unwound`]).

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};
use core::fmt;

use super::{Parser, ParserError};
use crate::ast::{DollarQuotedString, Ident, ObjectName, ObjectNamePart, QuoteDelimitedString};
use crate::funding::{Buffer, Funding, Held};
use crate::tokenizer::{
    Span, Token, TokenWithSpan, TokenizerError, TokenizerErrorKind, TokenizerFunding, Whitespace,
    Word,
};

impl<'a> Parser<'a> {
    /// Charge what the parser allocates to `funding` before it is allocated: an allocation it
    /// refuses (or the allocator refuses) fails the parse with [`ParserError::OutOfMemory`],
    /// nothing allocated for it.
    ///
    /// What the returned statements (or error) hold stays charged: [`Parser::kept`] is how
    /// much, to be given back to `funding` once they are freed.
    pub fn with_funding(mut self, funding: &'a dyn TokenizerFunding) -> Self {
        self.funding = Funding(funding);
        self
    }

    /// The bytes charged for what the parser made and handed out that are still held: what the
    /// statements, or the error, it returned hold. Whoever frees them gives these back to the
    /// funding.
    pub fn kept(&self) -> usize {
        self.kept.get()
    }

    /// One level deeper into the statement, refused once the recursion limit is reached.
    pub(crate) fn deeper<T>(
        &mut self,
        within: impl FnOnce(&mut Self) -> Result<T, ParserError>,
    ) -> Result<T, ParserError> {
        let Some(remaining) = self.remaining_depth.checked_sub(1) else {
            return Err(ParserError::RecursionLimitExceeded);
        };
        self.remaining_depth = remaining;
        let made = within(self);
        self.remaining_depth = remaining + 1;
        made
    }

    /// The buffer `held` made, handed out: its charge is kept with it.
    pub(crate) fn keep<B: Buffer>(&self, held: Held<'_, B>) -> B {
        self.kept.set(self.kept.get() + held.charged());
        held.keep()
    }

    /// An empty list, allocating nothing until it grows; kept with [`Parser::keep`].
    pub(crate) fn list<T>(&self) -> Held<'a, Vec<T>> {
        self.funding.list()
    }

    /// Exactly `items`, in a list handed out.
    pub(crate) fn vec_of<T, const N: usize>(&self, items: [T; N]) -> Result<Vec<T>, ParserError> {
        let made = self.funding.vec_of(items)?;
        Ok(self.keep(made))
    }

    /// Room for `additional` more items in a list (or text) already handed out: the block it
    /// grows into is charged before it is allocated, and kept in place of the one it grew out of.
    pub(crate) fn grow<B: Buffer>(
        &self,
        kept: &mut B,
        additional: usize,
    ) -> Result<(), ParserError> {
        let before = kept.room() * B::ITEM;
        self.funding.reserve(kept, additional)?;
        let grown = kept.room() * B::ITEM - before;
        self.kept.set(self.kept.get() + grown);
        Ok(())
    }

    /// `item` pushed onto a list already handed out, its growth charged first.
    pub(crate) fn push<T>(&self, kept: &mut Vec<T>, item: T) -> Result<(), ParserError> {
        self.grow(kept, 1)?;
        kept.push(item);
        Ok(())
    }

    /// Each of the items of `list` (a list handed out) handed to `each` in turn, stopping at the
    /// first that fails; the list's own block, freed then, is given back.
    pub(crate) fn each<T>(
        &self,
        list: Vec<T>,
        each: impl FnMut(T) -> Result<(), ParserError>,
    ) -> Result<(), ParserError> {
        let room = list.room() * <Vec<T> as Buffer>::ITEM;
        let made = list.into_iter().try_for_each(each);
        self.kept.set(self.kept.get() - room);
        self.funding.discharge(room);
        made
    }

    /// `kept`, a buffer handed out, freed: its room is given back. What its items hold is not.
    pub(crate) fn released<B: Buffer>(&self, kept: B) {
        let room = kept.room() * B::ITEM;
        drop(kept);
        self.kept.set(self.kept.get() - room);
        self.funding.discharge(room);
    }

    /// The only item of `list` (a list handed out), the list itself freed and given back.
    pub(crate) fn only<T>(&self, mut list: Vec<T>) -> Option<T> {
        let item = list.pop();
        self.released(list);
        item
    }

    /// `error`, made and handed out, freed: what it held is given back.
    pub(crate) fn discarded(&self, error: ParserError) {
        let held = error.held();
        drop(error);
        self.kept.set(self.kept.get() - held);
        self.funding.discharge(held);
    }

    /// The value `boxed` holds, moved out of its block: the block, freed, is given back.
    pub(crate) fn unboxed<T>(&self, boxed: Box<T>) -> T {
        let value = *boxed;
        let size = core::mem::size_of::<T>();
        self.kept.set(self.kept.get() - size);
        self.funding.discharge(size);
        value
    }

    /// The name `idents` (a list handed out) make, in a list of its own: the list of `idents`,
    /// freed, is given back.
    pub(crate) fn object_name(&self, idents: Vec<Ident>) -> Result<ObjectName, ParserError> {
        let mut parts = self.list();
        parts.reserve_exact(idents.len())?;
        self.each(idents, |ident| {
            parts
                .push(ObjectNamePart::Identifier(ident))
                .map_err(ParserError::from)
        })?;
        Ok(ObjectName(self.keep(parts)))
    }

    /// `value` in a block of its own, handed out.
    pub(crate) fn boxed<T>(&self, value: T) -> Result<Box<T>, ParserError> {
        let made = self.funding.boxed(value)?;
        self.kept.set(self.kept.get() + core::mem::size_of::<T>());
        Ok(made)
    }

    /// A copy of `text` in exactly its own bytes, handed out.
    pub(crate) fn string(&self, text: &str) -> Result<String, ParserError> {
        let made = self.funding.copy(text)?;
        Ok(self.keep(made))
    }

    /// `args` written out in exactly its bytes, handed out.
    pub(crate) fn format(&self, args: fmt::Arguments<'_>) -> Result<String, ParserError> {
        let made = self.funding.format(args)?;
        Ok(self.keep(made))
    }

    /// The syntax error `message`, the message charged before it is made: out of memory if that
    /// is refused.
    pub(crate) fn error(&self, message: fmt::Arguments<'_>) -> ParserError {
        match self.format(message) {
            Ok(message) => ParserError::ParserError(message),
            Err(refused) => refused,
        }
    }

    /// The parse error `error` is, its message charged before it is made: out of memory if
    /// that is refused.
    pub(crate) fn tokenizer_error(&self, error: &TokenizerError) -> ParserError {
        match error.kind {
            TokenizerErrorKind::OutOfMemory { needed } => ParserError::OutOfMemory { needed },
            TokenizerErrorKind::Syntax => match self.format(format_args!("{error}")) {
                Ok(message) => ParserError::TokenizerError(message),
                Err(refused) => refused,
            },
        }
    }

    /// `found` was met instead of what `expected` describes.
    pub(crate) fn expected_args(
        &self,
        expected: fmt::Arguments<'_>,
        found: &TokenWithSpan,
    ) -> ParserError {
        self.error(format_args!(
            "Expected: {expected}, found: {found}{}",
            found.span.start
        ))
    }

    /// The identifier `word` is, its text copied.
    pub(crate) fn ident(&self, word: &Word, span: Span) -> Result<Ident, ParserError> {
        Ok(Ident {
            value: self.string(&word.value)?,
            quote_style: word.quote_style,
            span,
        })
    }

    /// A copy of `token`, its text copied.
    pub(crate) fn copied(&self, token: &TokenWithSpan) -> Result<TokenWithSpan, ParserError> {
        Ok(TokenWithSpan {
            token: self.token(&token.token)?,
            span: token.span,
        })
    }

    /// A copy of `token`, its text copied.
    pub(crate) fn token(&self, token: &Token) -> Result<Token, ParserError> {
        Ok(match token {
            Token::Word(word) => Token::Word(Word {
                value: self.string(&word.value)?,
                quote_style: word.quote_style,
                keyword: word.keyword,
            }),
            Token::Number(text, long) => Token::Number(self.string(text)?, *long),
            Token::SingleQuotedString(text) => Token::SingleQuotedString(self.string(text)?),
            Token::DoubleQuotedString(text) => Token::DoubleQuotedString(self.string(text)?),
            Token::TripleSingleQuotedString(text) => {
                Token::TripleSingleQuotedString(self.string(text)?)
            }
            Token::TripleDoubleQuotedString(text) => {
                Token::TripleDoubleQuotedString(self.string(text)?)
            }
            Token::DollarQuotedString(quoted) => Token::DollarQuotedString(DollarQuotedString {
                value: self.string(&quoted.value)?,
                tag: match &quoted.tag {
                    Some(tag) => Some(self.string(tag)?),
                    None => None,
                },
            }),
            Token::SingleQuotedByteStringLiteral(text) => {
                Token::SingleQuotedByteStringLiteral(self.string(text)?)
            }
            Token::DoubleQuotedByteStringLiteral(text) => {
                Token::DoubleQuotedByteStringLiteral(self.string(text)?)
            }
            Token::TripleSingleQuotedByteStringLiteral(text) => {
                Token::TripleSingleQuotedByteStringLiteral(self.string(text)?)
            }
            Token::TripleDoubleQuotedByteStringLiteral(text) => {
                Token::TripleDoubleQuotedByteStringLiteral(self.string(text)?)
            }
            Token::SingleQuotedRawStringLiteral(text) => {
                Token::SingleQuotedRawStringLiteral(self.string(text)?)
            }
            Token::DoubleQuotedRawStringLiteral(text) => {
                Token::DoubleQuotedRawStringLiteral(self.string(text)?)
            }
            Token::TripleSingleQuotedRawStringLiteral(text) => {
                Token::TripleSingleQuotedRawStringLiteral(self.string(text)?)
            }
            Token::TripleDoubleQuotedRawStringLiteral(text) => {
                Token::TripleDoubleQuotedRawStringLiteral(self.string(text)?)
            }
            Token::NationalStringLiteral(text) => Token::NationalStringLiteral(self.string(text)?),
            Token::QuoteDelimitedStringLiteral(quoted) => {
                Token::QuoteDelimitedStringLiteral(self.quote_delimited(quoted)?)
            }
            Token::NationalQuoteDelimitedStringLiteral(quoted) => {
                Token::NationalQuoteDelimitedStringLiteral(self.quote_delimited(quoted)?)
            }
            Token::EscapedStringLiteral(text) => Token::EscapedStringLiteral(self.string(text)?),
            Token::UnicodeStringLiteral(text) => Token::UnicodeStringLiteral(self.string(text)?),
            Token::HexStringLiteral(text) => Token::HexStringLiteral(self.string(text)?),
            Token::Whitespace(Whitespace::SingleLineComment { comment, prefix }) => {
                Token::Whitespace(Whitespace::SingleLineComment {
                    comment: self.string(comment)?,
                    prefix: self.string(prefix)?,
                })
            }
            Token::Whitespace(Whitespace::MultiLineComment(text)) => {
                Token::Whitespace(Whitespace::MultiLineComment(self.string(text)?))
            }
            Token::Placeholder(text) => Token::Placeholder(self.string(text)?),
            Token::CustomBinaryOperator(text) => Token::CustomBinaryOperator(self.string(text)?),
            // Allocating nothing, these are copied as they are.
            Token::Whitespace(Whitespace::Space | Whitespace::Newline | Whitespace::Tab)
            | Token::Char(_)
            | Token::EOF
            | Token::Comma
            | Token::DoubleEq
            | Token::Eq
            | Token::Neq
            | Token::Lt
            | Token::Gt
            | Token::LtEq
            | Token::GtEq
            | Token::Spaceship
            | Token::Plus
            | Token::Minus
            | Token::Mul
            | Token::Div
            | Token::DuckIntDiv
            | Token::Mod
            | Token::StringConcat
            | Token::LParen
            | Token::RParen
            | Token::Period
            | Token::Colon
            | Token::DoubleColon
            | Token::Assignment
            | Token::SemiColon
            | Token::Backslash
            | Token::LBracket
            | Token::RBracket
            | Token::Ampersand
            | Token::Pipe
            | Token::Caret
            | Token::LBrace
            | Token::RBrace
            | Token::RArrow
            | Token::Sharp
            | Token::DoubleSharp
            | Token::Tilde
            | Token::TildeAsterisk
            | Token::ExclamationMarkTilde
            | Token::ExclamationMarkTildeAsterisk
            | Token::DoubleTilde
            | Token::DoubleTildeAsterisk
            | Token::ExclamationMarkDoubleTilde
            | Token::ExclamationMarkDoubleTildeAsterisk
            | Token::ShiftLeft
            | Token::ShiftRight
            | Token::Overlap
            | Token::ExclamationMark
            | Token::DoubleExclamationMark
            | Token::AtSign
            | Token::CaretAt
            | Token::PGSquareRoot
            | Token::PGCubeRoot
            | Token::Arrow
            | Token::LongArrow
            | Token::HashArrow
            | Token::AtDashAt
            | Token::QuestionMarkDash
            | Token::AmpersandLeftAngleBracket
            | Token::AmpersandRightAngleBracket
            | Token::AmpersandLeftAngleBracketVerticalBar
            | Token::VerticalBarAmpersandRightAngleBracket
            | Token::TwoWayArrow
            | Token::LeftAngleBracketCaret
            | Token::RightAngleBracketCaret
            | Token::QuestionMarkSharp
            | Token::QuestionMarkDashVerticalBar
            | Token::QuestionMarkDoubleVerticalBar
            | Token::TildeEqual
            | Token::ShiftLeftVerticalBar
            | Token::VerticalBarShiftRight
            | Token::VerticalBarRightAngleBracket
            | Token::HashLongArrow
            | Token::AtArrow
            | Token::ArrowAt
            | Token::HashMinus
            | Token::AtQuestion
            | Token::AtAt
            | Token::Question
            | Token::QuestionAnd
            | Token::QuestionPipe => token.clone(),
        })
    }

    pub(crate) fn quote_delimited(
        &self,
        quoted: &QuoteDelimitedString,
    ) -> Result<QuoteDelimitedString, ParserError> {
        Ok(QuoteDelimitedString {
            start_quote: quoted.start_quote,
            value: self.string(&quoted.value)?,
            end_quote: quoted.end_quote,
        })
    }

    /// What was made and handed out since [`Parser::kept`] was `before` has all been freed (the
    /// parser backtracked out of it), all but what `error` holds: given back.
    pub(crate) fn unwound(&self, before: usize, error: &ParserError) {
        let made = self.kept.get().saturating_sub(before);
        let held = error.held().min(made);
        self.funding.discharge(made - held);
        self.kept.set(before + held);
    }

    /// What was made and handed out since [`Parser::kept`] was `before` has all been freed:
    /// given back.
    pub(crate) fn dropped(&self, before: usize) {
        let made = self.kept.get().saturating_sub(before);
        self.funding.discharge(made);
        self.kept.set(before);
    }

    /// `value`, if any, in a block of its own, handed out.
    pub(crate) fn maybe_boxed<T>(&self, value: Option<T>) -> Result<Option<Box<T>>, ParserError> {
        match value {
            Some(value) => self.boxed(value).map(Some),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use super::{Parser, ParserError};
    use crate::dialect::GenericDialect;
    use crate::tokenizer::{TokenWithSpan, Tokenizer, TokenizerFunding};

    /// At most `limit` bytes charged at once; what the last charge it refused asked for, and
    /// the most it held, recorded.
    struct Limited {
        limit: usize,
        charged: Cell<usize>,
        peak: Cell<usize>,
        refused: Cell<usize>,
    }

    impl Limited {
        fn new(limit: usize) -> Self {
            Self {
                limit,
                charged: Cell::new(0),
                peak: Cell::new(0),
                refused: Cell::new(0),
            }
        }
    }

    impl TokenizerFunding for Limited {
        fn charge(&self, bytes: usize) -> bool {
            let charged = self.charged.get() + bytes;
            if charged > self.limit {
                self.refused.set(bytes);
                return false;
            }
            self.charged.set(charged);
            self.peak.set(self.peak.get().max(charged));
            true
        }

        fn discharge(&self, bytes: usize) {
            self.charged.set(self.charged.get() - bytes);
        }
    }

    /// `sql` tokenized, then parsed, both within `funding`: how many statements it holds.
    fn parsed(funding: &Limited, sql: &str) -> Result<usize, ParserError> {
        let dialect = GenericDialect {};
        let mut tokens: Vec<TokenWithSpan> = Vec::new();
        Tokenizer::new(&dialect, sql)
            .with_funding(funding)
            .tokenize_with_location_into_buf(&mut tokens)?;
        let mut parser = Parser::new(&dialect)
            .with_funding(funding)
            .with_tokens_with_locations(tokens);
        parser.parse_statements().map(|statements| statements.len())
    }

    #[test]
    fn a_refusal_carries_the_bytes_the_refused_charge_asked_for() {
        for sql in [
            "SELECT 'it''s', 1 WHERE false",
            "SELECT a, b FROM t WHERE x > 1",
            "CREATE VIEW v AS SELECT a FROM t",
        ] {
            let ample = Limited::new(usize::MAX);
            assert_eq!(parsed(&ample, sql), Ok(1), "{sql}");
            for limit in 0..ample.peak.get() {
                let funding = Limited::new(limit);
                let refused = parsed(&funding, sql);
                let needed = funding.refused.get();
                assert!(
                    needed > 0,
                    "{sql} within {limit} bytes: a charge was refused"
                );
                assert_eq!(
                    refused,
                    Err(ParserError::OutOfMemory { needed }),
                    "{sql} within {limit} bytes: refused for what the refused charge asked for"
                );
            }
        }
    }
}
