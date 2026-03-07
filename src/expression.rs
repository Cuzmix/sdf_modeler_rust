//! Minimal mathematical expression evaluator for light animation.
//!
//! Supports: float literals, `t` (time), `pi`, basic math ops (+, -, *, /),
//! unary negation, parentheses, and functions: sin, cos, abs, fract, floor,
//! clamp, mix, step, smoothstep.

use serde::{Deserialize, Serialize};

// ── Token ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f32),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;

    while pos < chars.len() {
        let ch = chars[pos];
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                pos += 1;
            }
            '+' => { tokens.push(Token::Plus); pos += 1; }
            '-' => { tokens.push(Token::Minus); pos += 1; }
            '*' => { tokens.push(Token::Star); pos += 1; }
            '/' => { tokens.push(Token::Slash); pos += 1; }
            '(' => { tokens.push(Token::LParen); pos += 1; }
            ')' => { tokens.push(Token::RParen); pos += 1; }
            ',' => { tokens.push(Token::Comma); pos += 1; }
            '0'..='9' | '.' => {
                let start = pos;
                while pos < chars.len() && (chars[pos].is_ascii_digit() || chars[pos] == '.') {
                    pos += 1;
                }
                let num_str: String = chars[start..pos].iter().collect();
                let value = num_str.parse::<f32>().map_err(|_| format!("Invalid number: {num_str}"))?;
                tokens.push(Token::Number(value));
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let start = pos;
                while pos < chars.len() && (chars[pos].is_ascii_alphanumeric() || chars[pos] == '_') {
                    pos += 1;
                }
                let ident: String = chars[start..pos].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            _ => return Err(format!("Unexpected character: '{ch}'")),
        }
    }
    Ok(tokens)
}

// ── AST ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    Literal(f32),
    Variable, // t
    Negate(Box<Expression>),
    BinOp {
        op: BinOpKind,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Call {
        function: FunctionKind,
        args: Vec<Expression>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FunctionKind {
    Sin,
    Cos,
    Abs,
    Fract,
    Floor,
    Clamp,
    Mix,
    Step,
    Smoothstep,
}

impl FunctionKind {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "sin" => Some(Self::Sin),
            "cos" => Some(Self::Cos),
            "abs" => Some(Self::Abs),
            "fract" => Some(Self::Fract),
            "floor" => Some(Self::Floor),
            "clamp" => Some(Self::Clamp),
            "mix" => Some(Self::Mix),
            "step" => Some(Self::Step),
            "smoothstep" => Some(Self::Smoothstep),
            _ => None,
        }
    }

    fn expected_args(self) -> usize {
        match self {
            Self::Sin | Self::Cos | Self::Abs | Self::Fract | Self::Floor => 1,
            Self::Step => 2,
            Self::Clamp | Self::Mix | Self::Smoothstep => 3,
        }
    }
}

// ── Parser (recursive descent) ─────────────────────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        token
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        match self.advance() {
            Some(ref tok) if tok == expected => Ok(()),
            Some(tok) => Err(format!("Expected {expected:?}, got {tok:?}")),
            None => Err(format!("Expected {expected:?}, got end of input")),
        }
    }

    /// expr = term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_term()?;
        while let Some(Token::Plus | Token::Minus) = self.peek() {
            let op_tok = self.advance().unwrap();
            let right = self.parse_term()?;
            let op = match op_tok {
                Token::Plus => BinOpKind::Add,
                Token::Minus => BinOpKind::Sub,
                _ => unreachable!(),
            };
            left = Expression::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// term = unary (('*' | '/') unary)*
    fn parse_term(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_unary()?;
        while let Some(Token::Star | Token::Slash) = self.peek() {
            let op_tok = self.advance().unwrap();
            let right = self.parse_unary()?;
            let op = match op_tok {
                Token::Star => BinOpKind::Mul,
                Token::Slash => BinOpKind::Div,
                _ => unreachable!(),
            };
            left = Expression::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// unary = '-' unary | primary
    fn parse_unary(&mut self) -> Result<Expression, String> {
        if let Some(Token::Minus) = self.peek() {
            self.advance();
            let inner = self.parse_unary()?;
            return Ok(Expression::Negate(Box::new(inner)));
        }
        self.parse_primary()
    }

    /// primary = NUMBER | 't' | 'pi' | function_call | '(' expr ')'
    fn parse_primary(&mut self) -> Result<Expression, String> {
        match self.advance() {
            Some(Token::Number(value)) => Ok(Expression::Literal(value)),
            Some(Token::Ident(name)) => {
                match name.as_str() {
                    "t" => Ok(Expression::Variable),
                    "pi" | "PI" => Ok(Expression::Literal(std::f32::consts::PI)),
                    _ => {
                        // Must be a function call
                        let func = FunctionKind::from_name(&name)
                            .ok_or_else(|| format!("Unknown function or variable: '{name}'"))?;
                        self.expect(&Token::LParen)?;
                        let mut args = Vec::new();
                        if self.peek() != Some(&Token::RParen) {
                            args.push(self.parse_expr()?);
                            while let Some(Token::Comma) = self.peek() {
                                self.advance();
                                args.push(self.parse_expr()?);
                            }
                        }
                        self.expect(&Token::RParen)?;
                        let expected = func.expected_args();
                        if args.len() != expected {
                            return Err(format!(
                                "{name}() expects {expected} argument(s), got {}",
                                args.len()
                            ));
                        }
                        Ok(Expression::Call { function: func, args })
                    }
                }
            }
            Some(Token::LParen) => {
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            Some(tok) => Err(format!("Unexpected token: {tok:?}")),
            None => Err("Unexpected end of input".to_string()),
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Parse a mathematical expression string into an AST.
pub fn parse_expression(input: &str) -> Result<Expression, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Empty expression".to_string());
    }
    let tokens = tokenize(trimmed)?;
    if tokens.is_empty() {
        return Err("Empty expression".to_string());
    }
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;
    if parser.pos < parser.tokens.len() {
        return Err(format!(
            "Unexpected token after expression: {:?}",
            parser.tokens[parser.pos]
        ));
    }
    Ok(expr)
}

/// Evaluate an expression at a given time value `t`.
pub fn evaluate(expr: &Expression, t: f32) -> f32 {
    match expr {
        Expression::Literal(value) => *value,
        Expression::Variable => t,
        Expression::Negate(inner) => -evaluate(inner, t),
        Expression::BinOp { op, left, right } => {
            let left_val = evaluate(left, t);
            let right_val = evaluate(right, t);
            match op {
                BinOpKind::Add => left_val + right_val,
                BinOpKind::Sub => left_val - right_val,
                BinOpKind::Mul => left_val * right_val,
                BinOpKind::Div => {
                    if right_val.abs() < 1e-10 {
                        0.0 // Division by zero returns 0
                    } else {
                        left_val / right_val
                    }
                }
            }
        }
        Expression::Call { function, args } => {
            let a = evaluate(&args[0], t);
            match function {
                FunctionKind::Sin => a.sin(),
                FunctionKind::Cos => a.cos(),
                FunctionKind::Abs => a.abs(),
                FunctionKind::Fract => a.fract(),
                FunctionKind::Floor => a.floor(),
                FunctionKind::Clamp => {
                    let min_val = evaluate(&args[1], t);
                    let max_val = evaluate(&args[2], t);
                    a.clamp(min_val, max_val)
                }
                FunctionKind::Mix => {
                    let b = evaluate(&args[1], t);
                    let factor = evaluate(&args[2], t);
                    a * (1.0 - factor) + b * factor
                }
                FunctionKind::Step => {
                    let x = evaluate(&args[1], t);
                    if x < a { 0.0 } else { 1.0 }
                }
                FunctionKind::Smoothstep => {
                    let edge1 = evaluate(&args[1], t);
                    let x = evaluate(&args[2], t);
                    let clamped = ((x - a) / (edge1 - a)).clamp(0.0, 1.0);
                    clamped * clamped * (3.0 - 2.0 * clamped)
                }
            }
        }
    }
}

/// Expression presets for common light animations.
pub struct ExpressionPreset {
    pub name: &'static str,
    pub intensity_expr: &'static str,
    pub color_hue_expr: &'static str,
}

pub const EXPRESSION_PRESETS: &[ExpressionPreset] = &[
    ExpressionPreset {
        name: "Pulse",
        intensity_expr: "0.5 + 0.5 * sin(t * 3.0)",
        color_hue_expr: "",
    },
    ExpressionPreset {
        name: "Flicker",
        intensity_expr: "0.8 + 0.2 * sin(t * 20.0) * sin(t * 7.3)",
        color_hue_expr: "",
    },
    ExpressionPreset {
        name: "Slow Glow",
        intensity_expr: "0.5 + 0.5 * sin(t * 0.5)",
        color_hue_expr: "",
    },
    ExpressionPreset {
        name: "Rainbow",
        intensity_expr: "",
        color_hue_expr: "fract(t * 0.1) * 360.0",
    },
];

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(input: &str, t: f32) -> f32 {
        let expr = parse_expression(input).unwrap();
        evaluate(&expr, t)
    }

    #[test]
    fn parse_literal() {
        assert!((eval("42.0", 0.0) - 42.0).abs() < 1e-5);
    }

    #[test]
    fn parse_variable_t() {
        assert!((eval("t", 3.5) - 3.5).abs() < 1e-5);
    }

    #[test]
    fn parse_pi_constant() {
        assert!((eval("pi", 0.0) - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn basic_arithmetic() {
        assert!((eval("2.0 + 3.0", 0.0) - 5.0).abs() < 1e-5);
        assert!((eval("10.0 - 4.0", 0.0) - 6.0).abs() < 1e-5);
        assert!((eval("3.0 * 4.0", 0.0) - 12.0).abs() < 1e-5);
        assert!((eval("10.0 / 4.0", 0.0) - 2.5).abs() < 1e-5);
    }

    #[test]
    fn operator_precedence() {
        assert!((eval("2.0 + 3.0 * 4.0", 0.0) - 14.0).abs() < 1e-5);
        assert!((eval("(2.0 + 3.0) * 4.0", 0.0) - 20.0).abs() < 1e-5);
    }

    #[test]
    fn unary_negation() {
        assert!((eval("-5.0", 0.0) - (-5.0)).abs() < 1e-5);
        assert!((eval("-t", 3.0) - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn sin_cos_at_known_values() {
        assert!((eval("sin(0.0)", 0.0) - 0.0).abs() < 1e-5);
        assert!((eval("cos(0.0)", 0.0) - 1.0).abs() < 1e-5);
        let half_pi = std::f32::consts::FRAC_PI_2;
        assert!((eval("sin(pi / 2.0)", 0.0) - half_pi.sin()).abs() < 1e-4);
    }

    #[test]
    fn abs_function() {
        assert!((eval("abs(-7.0)", 0.0) - 7.0).abs() < 1e-5);
        assert!((eval("abs(3.0)", 0.0) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn fract_function() {
        assert!((eval("fract(3.7)", 0.0) - 0.7).abs() < 1e-4);
    }

    #[test]
    fn floor_function() {
        assert!((eval("floor(3.7)", 0.0) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn clamp_function() {
        assert!((eval("clamp(5.0, 0.0, 1.0)", 0.0) - 1.0).abs() < 1e-5);
        assert!((eval("clamp(-1.0, 0.0, 1.0)", 0.0) - 0.0).abs() < 1e-5);
        assert!((eval("clamp(0.5, 0.0, 1.0)", 0.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn mix_function() {
        assert!((eval("mix(0.0, 10.0, 0.5)", 0.0) - 5.0).abs() < 1e-5);
        assert!((eval("mix(0.0, 10.0, 0.0)", 0.0) - 0.0).abs() < 1e-5);
        assert!((eval("mix(0.0, 10.0, 1.0)", 0.0) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn step_function() {
        assert!((eval("step(0.5, 0.3)", 0.0) - 0.0).abs() < 1e-5);
        assert!((eval("step(0.5, 0.7)", 0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn smoothstep_function() {
        assert!((eval("smoothstep(0.0, 1.0, 0.5)", 0.0) - 0.5).abs() < 0.1);
        assert!((eval("smoothstep(0.0, 1.0, 0.0)", 0.0) - 0.0).abs() < 1e-5);
        assert!((eval("smoothstep(0.0, 1.0, 1.0)", 0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn division_by_zero_returns_zero() {
        assert!((eval("1.0 / 0.0", 0.0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn empty_expression_returns_error() {
        assert!(parse_expression("").is_err());
        assert!(parse_expression("   ").is_err());
    }

    #[test]
    fn unknown_function_returns_error() {
        assert!(parse_expression("bogus(1.0)").is_err());
    }

    #[test]
    fn wrong_arg_count_returns_error() {
        assert!(parse_expression("sin(1.0, 2.0)").is_err());
        assert!(parse_expression("clamp(1.0)").is_err());
    }

    #[test]
    fn complex_expression_evaluates() {
        // Pulse preset: 0.5 + 0.5 * sin(t * 3.0)
        let result = eval("0.5 + 0.5 * sin(t * 3.0)", 0.0);
        assert!((result - 0.5).abs() < 1e-5); // sin(0) = 0

        // At t=pi/(2*3), sin(pi/2) = 1 → 0.5 + 0.5 = 1.0
        let t_val = std::f32::consts::FRAC_PI_2 / 3.0;
        let result = eval("0.5 + 0.5 * sin(t * 3.0)", t_val);
        assert!((result - 1.0).abs() < 1e-4);
    }

    #[test]
    fn flicker_preset_evaluates() {
        let result = eval("0.8 + 0.2 * sin(t * 20.0) * sin(t * 7.3)", 0.0);
        assert!((result - 0.8).abs() < 1e-5); // sin(0)*sin(0) = 0
    }

    #[test]
    fn rainbow_preset_evaluates() {
        let result = eval("fract(t * 0.1) * 360.0", 0.0);
        assert!((result - 0.0).abs() < 1e-5);

        let result = eval("fract(t * 0.1) * 360.0", 5.0);
        assert!((result - 180.0).abs() < 1e-3);
    }

    #[test]
    fn nested_parentheses() {
        assert!((eval("((2.0 + 3.0) * (4.0 - 1.0))", 0.0) - 15.0).abs() < 1e-5);
    }

    #[test]
    fn expression_with_t_and_operations() {
        assert!((eval("t * t + 1.0", 3.0) - 10.0).abs() < 1e-5);
    }

    // ── t=0 produces expected results for common functions ──────────

    #[test]
    fn sin_at_t_zero_is_zero() {
        assert!(eval("sin(t)", 0.0).abs() < 1e-5);
    }

    #[test]
    fn cos_at_t_zero_is_one() {
        assert!((eval("cos(t)", 0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fract_at_t_zero_is_zero() {
        assert!(eval("fract(t)", 0.0).abs() < 1e-5);
    }

    #[test]
    fn pulse_preset_at_t_zero() {
        // "0.5 + 0.5 * sin(t * 3.0)" at t=0 → 0.5 + 0.5*sin(0) = 0.5
        assert!((eval("0.5 + 0.5 * sin(t * 3.0)", 0.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn slow_glow_preset_at_t_zero() {
        // "0.5 + 0.5 * sin(t * 0.5)" at t=0 → 0.5
        assert!((eval("0.5 + 0.5 * sin(t * 0.5)", 0.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn rainbow_preset_at_t_zero() {
        // "fract(t * 0.1) * 360.0" at t=0 → 0
        assert!(eval("fract(t * 0.1) * 360.0", 0.0).abs() < 1e-5);
    }

    #[test]
    fn flicker_preset_at_t_zero() {
        // "0.8 + 0.2 * sin(t * 20.0) * sin(t * 7.3)" at t=0 → 0.8 + 0.2*0*0 = 0.8
        assert!((eval("0.8 + 0.2 * sin(t * 20.0) * sin(t * 7.3)", 0.0) - 0.8).abs() < 1e-5);
    }
}
