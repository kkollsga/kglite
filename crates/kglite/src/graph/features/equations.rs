// src/graph/equations.rs
use crate::datatypes::values::Value;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Expr {
    Number(f64),
    Variable(String),
    Add(Box<Expr>, Box<Expr>),
    Subtract(Box<Expr>, Box<Expr>),
    Multiply(Box<Expr>, Box<Expr>),
    Divide(Box<Expr>, Box<Expr>),
    Aggregate(AggregateType, Box<Expr>),
}

impl Expr {
    pub fn extract_variables(&self) -> Vec<String> {
        let mut variables = Vec::new();
        self.collect_variables(&mut variables);
        variables.sort();
        variables.dedup();
        variables
    }

    fn collect_variables(&self, variables: &mut Vec<String>) {
        match self {
            Expr::Variable(name) => variables.push(name.clone()),
            Expr::Add(left, right)
            | Expr::Subtract(left, right)
            | Expr::Multiply(left, right)
            | Expr::Divide(left, right) => {
                left.collect_variables(variables);
                right.collect_variables(variables);
            }
            Expr::Aggregate(_, inner) => inner.collect_variables(variables),
            Expr::Number(_) => {} // Number doesn't contain variables
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggregateType {
    Sum,
    Mean,
    Std,
    Min,
    Max,
    Count,
}

impl AggregateType {
    // Helper function to get all supported aggregate function names
    pub fn get_supported_names() -> Vec<&'static str> {
        vec!["sum", "mean", "avg", "std", "min", "max", "count"]
    }

    // Helper function to convert from string
    pub fn from_string(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "sum" => Some(AggregateType::Sum),
            "mean" | "avg" | "average" => Some(AggregateType::Mean),
            "std" => Some(AggregateType::Std),
            "min" => Some(AggregateType::Min),
            "max" => Some(AggregateType::Max),
            "count" => Some(AggregateType::Count),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Identifier(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Aggregate(AggregateType),
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn parse_expression(input: &str) -> Result<Expr, String> {
        let mut parser = Self::new(input);
        parser.parse_aggregate()
    }

    fn new(input: &str) -> Self {
        let tokens = Self::tokenize(input);
        Parser { tokens, pos: 0 }
    }

    fn tokenize(input: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut chars = input.chars().peekable();

        while let Some(&c) = chars.peek() {
            match c {
                '0'..='9' | '.' => {
                    let mut number = String::new();
                    let mut has_decimal = false;

                    while let Some(&c) = chars.peek() {
                        match c {
                            '0'..='9' => {
                                number.push(c);
                                chars.next();
                            }
                            '.' if !has_decimal => {
                                has_decimal = true;
                                number.push(c);
                                chars.next();
                            }
                            _ => break,
                        }
                    }

                    if let Ok(num) = number.parse() {
                        tokens.push(Token::Number(num));
                    }
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    let mut ident = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_alphanumeric() || c == '_' {
                            ident.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }

                    // Check if the identifier is an aggregate function
                    if let Some(agg_type) = AggregateType::from_string(&ident) {
                        tokens.push(Token::Aggregate(agg_type));
                    } else {
                        tokens.push(Token::Identifier(ident));
                    }
                }
                '+' => {
                    tokens.push(Token::Plus);
                    chars.next();
                }
                '-' => {
                    tokens.push(Token::Minus);
                    chars.next();
                }
                '*' => {
                    tokens.push(Token::Star);
                    chars.next();
                }
                '/' => {
                    tokens.push(Token::Slash);
                    chars.next();
                }
                '(' => {
                    tokens.push(Token::LParen);
                    chars.next();
                }
                ')' => {
                    tokens.push(Token::RParen);
                    chars.next();
                }
                ' ' | '\t' | '\n' | '\r' => {
                    chars.next();
                }
                _ => return vec![], // Return empty vec for invalid input
            }
        }
        tokens
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn consume(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            self.pos += 1;
            Some(self.tokens[self.pos - 1].clone())
        } else {
            None
        }
    }

    fn peek_and_consume_if<F>(&mut self, predicate: F) -> Option<Token>
    where
        F: FnOnce(&Token) -> bool,
    {
        match self.peek() {
            Some(token) if predicate(token) => self.consume(),
            _ => None,
        }
    }

    fn parse_aggregate(&mut self) -> Result<Expr, String> {
        let token = self.peek_and_consume_if(|t| matches!(t, Token::Aggregate(_)));

        match token {
            Some(Token::Aggregate(agg_type)) => {
                // Expect opening parenthesis
                if self
                    .peek_and_consume_if(|t| matches!(t, Token::LParen))
                    .is_none()
                {
                    return Err("Expected opening parenthesis after aggregate function".to_string());
                }

                // Parse the inner expression
                let expr = self.parse_expr()?;

                // Expect closing parenthesis
                if self
                    .peek_and_consume_if(|t| matches!(t, Token::RParen))
                    .is_none()
                {
                    return Err(
                        "Expected closing parenthesis after aggregate expression".to_string()
                    );
                }

                Ok(Expr::Aggregate(agg_type, Box::new(expr)))
            }
            _ => self.parse_expr(),
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_term()?;

        while let Some(token) = self.peek() {
            match *token {
                Token::Plus => {
                    self.consume();
                    let right = self.parse_term()?;
                    left = Expr::Add(Box::new(left), Box::new(right));
                }
                Token::Minus => {
                    self.consume();
                    let right = self.parse_term()?;
                    left = Expr::Subtract(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_factor()?;

        while let Some(token) = self.peek() {
            match *token {
                Token::Star => {
                    self.consume();
                    let right = self.parse_factor()?;
                    left = Expr::Multiply(Box::new(left), Box::new(right));
                }
                Token::Slash => {
                    self.consume();
                    let right = self.parse_factor()?;
                    left = Expr::Divide(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_factor(&mut self) -> Result<Expr, String> {
        let token = self.peek().cloned();
        match token {
            Some(Token::Number(n)) => {
                self.consume();
                Ok(Expr::Number(n))
            }
            Some(Token::Identifier(name)) => {
                self.consume();
                Ok(Expr::Variable(name))
            }
            Some(Token::LParen) => {
                self.consume();
                let expr = self.parse_expr()?;
                if self
                    .peek_and_consume_if(|t| matches!(t, Token::RParen))
                    .is_none()
                {
                    return Err("Expected closing parenthesis".to_string());
                }
                Ok(expr)
            }
            None => Err("Unexpected end of expression".to_string()),
            _ => Err("Unexpected token in factor".to_string()),
        }
    }
}

pub struct Evaluator;

impl Evaluator {
    pub fn evaluate(expr: &Expr, objects: &[HashMap<String, Value>]) -> Result<Value, String> {
        match expr {
            Expr::Aggregate(agg_type, inner) => {
                // Track results and errors for debugging
                let mut total_objects = 0;
                let mut successful_evals = 0;
                let mut null_results = 0;
                let mut error_messages = Vec::new();

                let values: Vec<f64> = objects
                    .iter()
                    .filter_map(|obj| {
                        total_objects += 1;
                        match Self::evaluate_single(inner, obj) {
                            Ok(value) => {
                                if matches!(value, Value::Null) {
                                    null_results += 1;
                                    None
                                } else if let Some(f) =
                                    crate::graph::core::value_operations::value_to_f64(&value)
                                {
                                    successful_evals += 1;
                                    Some(f)
                                } else {
                                    None
                                }
                            }
                            Err(msg) => {
                                error_messages.push(msg);
                                None
                            }
                        }
                    })
                    .collect();

                // If we have no valid evaluations but have objects, return 0 for sum operations
                // instead of null - this makes sums more intuitive when fields are missing
                if total_objects > 0 && successful_evals == 0 {
                    if !error_messages.is_empty() {
                        // Print just a few error messages to avoid flooding the output
                        let sample_errors: Vec<_> = error_messages.iter().take(3).collect();
                        println!(
                            "Warning: All evaluations failed. Sample errors: {:?}",
                            sample_errors
                        );
                    }

                    // For sum and count, return 0 when all values are null or errors
                    // For other aggregates, return null
                    return match agg_type {
                        AggregateType::Sum | AggregateType::Count => Ok(Value::Float64(0.0)),
                        _ => Ok(Value::Null),
                    };
                }

                if values.is_empty() {
                    return match agg_type {
                        AggregateType::Sum | AggregateType::Count => Ok(Value::Float64(0.0)),
                        _ => Ok(Value::Null),
                    };
                }

                // Use shared aggregation functions (population std: divides by N)
                let result = match agg_type {
                    AggregateType::Sum => {
                        crate::graph::core::value_operations::aggregate_sum(&values)
                    }
                    AggregateType::Mean => {
                        crate::graph::core::value_operations::aggregate_mean(&values).unwrap()
                    }
                    AggregateType::Std => {
                        crate::graph::core::value_operations::aggregate_std(&values, true).unwrap()
                    }
                    AggregateType::Min => {
                        crate::graph::core::value_operations::aggregate_min(&values).unwrap()
                    }
                    AggregateType::Max => {
                        crate::graph::core::value_operations::aggregate_max(&values).unwrap()
                    }
                    AggregateType::Count => values.len() as f64,
                };

                Ok(Value::Float64(result))
            }
            // The rest of the code for non-aggregate expressions
            _ => {
                if objects.len() == 1 {
                    Self::evaluate_single(expr, &objects[0])
                } else {
                    Err("Expected single object for non-aggregate expression".to_string())
                }
            }
        }
    }

    fn evaluate_single(expr: &Expr, object: &HashMap<String, Value>) -> Result<Value, String> {
        match expr {
            Expr::Number(n) => Ok(Value::Float64(*n)),
            Expr::Variable(name) => {
                // Return null instead of error for missing variables
                Ok(object.get(name).cloned().unwrap_or(Value::Null))
            }
            Expr::Add(left, right) => {
                match (
                    Self::evaluate_single(left, object)?,
                    Self::evaluate_single(right, object)?,
                ) {
                    (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a + b)),
                    (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a + b)),
                    (Value::Int64(a), Value::Float64(b)) => Ok(Value::Float64(a as f64 + b)),
                    (Value::Float64(a), Value::Int64(b)) => Ok(Value::Float64(a + b as f64)),
                    (Value::UniqueId(a), Value::UniqueId(b)) => Ok(Value::UniqueId(a + b)),
                    (Value::Null, Value::Null) => Ok(Value::Null),
                    // Treat null as zero in additions for more lenient calculations
                    (Value::Null, Value::Int64(b)) => Ok(Value::Int64(b)),
                    (Value::Int64(a), Value::Null) => Ok(Value::Int64(a)),
                    (Value::Null, Value::Float64(b)) => Ok(Value::Float64(b)),
                    (Value::Float64(a), Value::Null) => Ok(Value::Float64(a)),
                    (Value::Null, Value::UniqueId(b)) => Ok(Value::UniqueId(b)),
                    (Value::UniqueId(a), Value::Null) => Ok(Value::UniqueId(a)),
                    (a, b) => Err(format!("Cannot add values: {:?} and {:?}", a, b)),
                }
            }
            Expr::Subtract(left, right) => {
                match (
                    Self::evaluate_single(left, object)?,
                    Self::evaluate_single(right, object)?,
                ) {
                    (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a - b)),
                    (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a - b)),
                    (Value::Int64(a), Value::Float64(b)) => Ok(Value::Float64(a as f64 - b)),
                    (Value::Float64(a), Value::Int64(b)) => Ok(Value::Float64(a - b as f64)),
                    (Value::UniqueId(a), Value::UniqueId(b)) => Ok(Value::UniqueId(a - b)),
                    (Value::Null, Value::Null) => Ok(Value::Null),
                    // Treat null specially in subtractions
                    (Value::Null, _) => Ok(Value::Null), // null - anything = null
                    (Value::Int64(a), Value::Null) => Ok(Value::Int64(a)), // a - null = a
                    (Value::Float64(a), Value::Null) => Ok(Value::Float64(a)),
                    (Value::UniqueId(a), Value::Null) => Ok(Value::UniqueId(a)),
                    (a, b) => Err(format!("Cannot subtract values: {:?} and {:?}", a, b)),
                }
            }
            Expr::Multiply(left, right) => {
                match (
                    Self::evaluate_single(left, object)?,
                    Self::evaluate_single(right, object)?,
                ) {
                    (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a * b)),
                    (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a * b)),
                    (Value::Int64(a), Value::Float64(b)) => Ok(Value::Float64(a as f64 * b)),
                    (Value::Float64(a), Value::Int64(b)) => Ok(Value::Float64(a * b as f64)),
                    (Value::UniqueId(a), Value::UniqueId(b)) => Ok(Value::UniqueId(a * b)),
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null), // null * anything = null
                    (a, b) => Err(format!("Cannot multiply values: {:?} and {:?}", a, b)),
                }
            }
            Expr::Divide(left, right) => {
                match (
                    Self::evaluate_single(left, object)?,
                    Self::evaluate_single(right, object)?,
                ) {
                    (_, Value::Int64(0)) | (_, Value::Float64(0.0)) | (_, Value::UniqueId(0)) => {
                        Ok(Value::Null) // Return null for division by zero
                    }
                    (Value::Int64(a), Value::Int64(b)) => Ok(Value::Float64(a as f64 / b as f64)),
                    (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a / b)),
                    (Value::Int64(a), Value::Float64(b)) => Ok(Value::Float64(a as f64 / b)),
                    (Value::Float64(a), Value::Int64(b)) => Ok(Value::Float64(a / b as f64)),
                    (Value::UniqueId(a), Value::UniqueId(b)) => {
                        Ok(Value::Float64(a as f64 / b as f64))
                    }
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null), // null / anything = null
                    (a, b) => Err(format!("Cannot divide values: {:?} and {:?}", a, b)),
                }
            }
            Expr::Aggregate(_, _) => Err("Nested aggregates not supported".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::values::Value;
    use std::collections::HashMap;

    // ========================================================================
    // AggregateType::from_string
    // ========================================================================

    #[test]
    fn test_aggregate_type_from_string() {
        assert_eq!(AggregateType::from_string("sum"), Some(AggregateType::Sum));
        assert_eq!(
            AggregateType::from_string("mean"),
            Some(AggregateType::Mean)
        );
        assert_eq!(AggregateType::from_string("avg"), Some(AggregateType::Mean));
        assert_eq!(
            AggregateType::from_string("average"),
            Some(AggregateType::Mean)
        );
        assert_eq!(AggregateType::from_string("std"), Some(AggregateType::Std));
        assert_eq!(AggregateType::from_string("min"), Some(AggregateType::Min));
        assert_eq!(AggregateType::from_string("max"), Some(AggregateType::Max));
        assert_eq!(
            AggregateType::from_string("count"),
            Some(AggregateType::Count)
        );
        assert_eq!(AggregateType::from_string("unknown"), None);
    }

    #[test]
    fn test_aggregate_type_case_insensitive() {
        assert_eq!(AggregateType::from_string("SUM"), Some(AggregateType::Sum));
        assert_eq!(
            AggregateType::from_string("Mean"),
            Some(AggregateType::Mean)
        );
    }

    // ========================================================================
    // Parser — simple expressions
    // ========================================================================

    #[test]
    fn test_parse_number() {
        let expr = Parser::parse_expression("42").unwrap();
        assert!(matches!(expr, Expr::Number(n) if (n - 42.0).abs() < f64::EPSILON));
    }

    #[test]
    fn test_parse_variable() {
        let expr = Parser::parse_expression("price").unwrap();
        assert!(matches!(expr, Expr::Variable(ref name) if name == "price"));
    }

    #[test]
    fn test_parse_addition() {
        let expr = Parser::parse_expression("a + b").unwrap();
        assert!(matches!(expr, Expr::Add(_, _)));
    }

    #[test]
    fn test_parse_subtraction() {
        let expr = Parser::parse_expression("a - b").unwrap();
        assert!(matches!(expr, Expr::Subtract(_, _)));
    }

    #[test]
    fn test_parse_multiplication() {
        let expr = Parser::parse_expression("a * b").unwrap();
        assert!(matches!(expr, Expr::Multiply(_, _)));
    }

    #[test]
    fn test_parse_division() {
        let expr = Parser::parse_expression("a / b").unwrap();
        assert!(matches!(expr, Expr::Divide(_, _)));
    }

    #[test]
    fn test_parse_operator_precedence() {
        // a + b * c should be a + (b * c)
        let expr = Parser::parse_expression("a + b * c").unwrap();
        match expr {
            Expr::Add(_, right) => assert!(matches!(*right, Expr::Multiply(_, _))),
            _ => panic!("Expected Add at top level"),
        }
    }

    #[test]
    fn test_parse_parentheses() {
        // (a + b) * c should be (a + b) * c
        let expr = Parser::parse_expression("(a + b) * c").unwrap();
        match expr {
            Expr::Multiply(left, _) => assert!(matches!(*left, Expr::Add(_, _))),
            _ => panic!("Expected Multiply at top level"),
        }
    }

    #[test]
    fn test_parse_aggregate() {
        let expr = Parser::parse_expression("sum(value)").unwrap();
        match expr {
            Expr::Aggregate(AggregateType::Sum, inner) => {
                assert!(matches!(*inner, Expr::Variable(ref name) if name == "value"));
            }
            _ => panic!("Expected Aggregate Sum"),
        }
    }

    // ========================================================================
    // Expr::extract_variables
    // ========================================================================

    #[test]
    fn test_extract_variables() {
        let expr = Parser::parse_expression("a + b * c").unwrap();
        let vars = expr.extract_variables();
        assert_eq!(vars, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_extract_variables_deduplicates() {
        let expr = Parser::parse_expression("a + a").unwrap();
        let vars = expr.extract_variables();
        assert_eq!(vars, vec!["a"]);
    }

    #[test]
    fn test_extract_variables_number_has_none() {
        let expr = Parser::parse_expression("42").unwrap();
        let vars = expr.extract_variables();
        assert!(vars.is_empty());
    }

    // ========================================================================
    // Evaluator — single-object evaluation
    // ========================================================================

    #[test]
    fn test_evaluate_number() {
        let expr = Parser::parse_expression("42").unwrap();
        let objs = vec![HashMap::new()];
        let result = Evaluator::evaluate(&expr, &objs).unwrap();
        assert_eq!(result, Value::Float64(42.0));
    }

    #[test]
    fn test_evaluate_variable() {
        let expr = Parser::parse_expression("price").unwrap();
        let mut obj = HashMap::new();
        obj.insert("price".to_string(), Value::Float64(9.99));
        let result = Evaluator::evaluate(&expr, &[obj]).unwrap();
        assert_eq!(result, Value::Float64(9.99));
    }

    #[test]
    fn test_evaluate_missing_variable_returns_null() {
        let expr = Parser::parse_expression("missing").unwrap();
        let objs = vec![HashMap::new()];
        let result = Evaluator::evaluate(&expr, &objs).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_evaluate_arithmetic() {
        let expr = Parser::parse_expression("a + b").unwrap();
        let mut obj = HashMap::new();
        obj.insert("a".to_string(), Value::Int64(3));
        obj.insert("b".to_string(), Value::Int64(7));
        let result = Evaluator::evaluate(&expr, &[obj]).unwrap();
        assert_eq!(result, Value::Int64(10));
    }

    #[test]
    fn test_evaluate_division_by_zero_returns_null() {
        let expr = Parser::parse_expression("a / b").unwrap();
        let mut obj = HashMap::new();
        obj.insert("a".to_string(), Value::Float64(10.0));
        obj.insert("b".to_string(), Value::Float64(0.0));
        let result = Evaluator::evaluate(&expr, &[obj]).unwrap();
        assert_eq!(result, Value::Null);
    }

    // ========================================================================
    // Evaluator — aggregation
    // ========================================================================

    #[test]
    fn test_evaluate_sum() {
        let expr = Parser::parse_expression("sum(value)").unwrap();
        let objs: Vec<HashMap<String, Value>> = (1..=5)
            .map(|i| {
                let mut m = HashMap::new();
                m.insert("value".to_string(), Value::Float64(i as f64));
                m
            })
            .collect();
        let result = Evaluator::evaluate(&expr, &objs).unwrap();
        assert_eq!(result, Value::Float64(15.0));
    }

    #[test]
    fn test_evaluate_mean() {
        let expr = Parser::parse_expression("mean(value)").unwrap();
        let objs: Vec<HashMap<String, Value>> = vec![2.0, 4.0, 6.0]
            .into_iter()
            .map(|v| {
                let mut m = HashMap::new();
                m.insert("value".to_string(), Value::Float64(v));
                m
            })
            .collect();
        let result = Evaluator::evaluate(&expr, &objs).unwrap();
        assert_eq!(result, Value::Float64(4.0));
    }

    #[test]
    fn test_evaluate_count() {
        let expr = Parser::parse_expression("count(value)").unwrap();
        let objs: Vec<HashMap<String, Value>> = (0..3)
            .map(|i| {
                let mut m = HashMap::new();
                m.insert("value".to_string(), Value::Int64(i));
                m
            })
            .collect();
        let result = Evaluator::evaluate(&expr, &objs).unwrap();
        assert_eq!(result, Value::Float64(3.0));
    }

    #[test]
    fn test_evaluate_min_max() {
        let expr_min = Parser::parse_expression("min(v)").unwrap();
        let expr_max = Parser::parse_expression("max(v)").unwrap();
        let objs: Vec<HashMap<String, Value>> = vec![5.0, 1.0, 9.0, 3.0]
            .into_iter()
            .map(|v| {
                let mut m = HashMap::new();
                m.insert("v".to_string(), Value::Float64(v));
                m
            })
            .collect();
        assert_eq!(
            Evaluator::evaluate(&expr_min, &objs).unwrap(),
            Value::Float64(1.0)
        );
        assert_eq!(
            Evaluator::evaluate(&expr_max, &objs).unwrap(),
            Value::Float64(9.0)
        );
    }

    #[test]
    fn test_evaluate_sum_empty_returns_zero() {
        let expr = Parser::parse_expression("sum(value)").unwrap();
        let objs: Vec<HashMap<String, Value>> = vec![];
        let result = Evaluator::evaluate(&expr, &objs).unwrap();
        assert_eq!(result, Value::Float64(0.0));
    }

    #[test]
    fn test_evaluate_mean_empty_returns_null() {
        let expr = Parser::parse_expression("mean(value)").unwrap();
        let objs: Vec<HashMap<String, Value>> = vec![];
        let result = Evaluator::evaluate(&expr, &objs).unwrap();
        assert_eq!(result, Value::Null);
    }
}
