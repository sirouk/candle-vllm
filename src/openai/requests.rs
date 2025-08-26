use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Messages {
    Map(Vec<HashMap<String, String>>),
    Literal(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopTokens {
    Multi(Vec<String>),
    Single(String),
}

/// Custom deserializer for logprobs that accepts only boolean values
#[derive(Debug, Clone, Default)]
pub struct LogprobsField(pub Option<usize>);

impl<'de> Deserialize<'de> for LogprobsField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Visitor};
        use std::fmt;

        struct LogprobsVisitor;

        impl<'de> Visitor<'de> for LogprobsVisitor {
            type Value = LogprobsField;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("boolean")
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if v {
                    // When logprobs: true, only return basic logprob (no top_logprobs)
                    Ok(LogprobsField(Some(0)))
                } else {
                    // When logprobs: false, no logprobs
                    Ok(LogprobsField(None))
                }
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(LogprobsField(None))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(LogprobsField(None))
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                Deserialize::deserialize(deserializer)
            }
        }

        deserializer.deserialize_bool(LogprobsVisitor)
    }
}

impl Serialize for LogprobsField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.0 {
            Some(_) => serializer.serialize_bool(true),
            None => serializer.serialize_bool(false),
        }
    }
}

impl From<LogprobsField> for Option<usize> {
    fn from(field: LogprobsField) -> Self {
        field.0
    }
}

impl From<Option<usize>> for LogprobsField {
    fn from(opt: Option<usize>) -> Self {
        LogprobsField(opt)
    }
}

/// Custom deserializer for top_logprobs that accepts boolean or number values
#[derive(Debug, Clone, Default)]
pub struct TopLogprobsField(pub Option<usize>);

impl<'de> Deserialize<'de> for TopLogprobsField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Visitor};
        use std::fmt;

        struct TopLogprobsVisitor;

        impl<'de> Visitor<'de> for TopLogprobsVisitor {
            type Value = TopLogprobsField;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("boolean or number")
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if v {
                    // When top_logprobs: true, default to 5 top logprobs
                    Ok(TopLogprobsField(Some(5)))
                } else {
                    // When top_logprobs: false, return empty array (None)
                    Ok(TopLogprobsField(None))
                }
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                // When top_logprobs: N, return N top logprobs
                Ok(TopLogprobsField(Some(v as usize)))
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if v < 0 {
                    return Err(de::Error::invalid_value(de::Unexpected::Signed(v), &"non-negative number"));
                }
                // When top_logprobs: N, return N top logprobs
                Ok(TopLogprobsField(Some(v as usize)))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(TopLogprobsField(None))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(TopLogprobsField(None))
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                Deserialize::deserialize(deserializer)
            }
        }

        deserializer.deserialize_any(TopLogprobsVisitor)
    }
}

impl Serialize for TopLogprobsField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.0 {
            Some(n) => serializer.serialize_u64(n as u64),
            None => serializer.serialize_bool(false),
        }
    }
}

impl From<TopLogprobsField> for Option<usize> {
    fn from(field: TopLogprobsField) -> Self {
        field.0
    }
}

impl From<Option<usize>> for TopLogprobsField {
    fn from(opt: Option<usize>) -> Self {
        TopLogprobsField(opt)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Messages,
    pub temperature: Option<f32>, //0.7
    pub top_p: Option<f32>,       //1.0
    #[serde(default)]
    pub n: Option<usize>, //1
    pub max_tokens: Option<usize>, //None
    #[serde(default)]
    pub stop: Option<StopTokens>,
    #[serde(default)]
    pub stream: Option<bool>, //false
    #[serde(default)]
    pub presence_penalty: Option<f32>, //0.0
    pub repeat_last_n: Option<usize>, //0.0
    #[serde(default)]
    pub frequency_penalty: Option<f32>, //0.0
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>, //None
    #[serde(default)]
    pub user: Option<String>, //None
    pub top_k: Option<isize>,         //-1
    #[serde(default)]
    pub best_of: Option<usize>, //None
    #[serde(default)]
    pub use_beam_search: Option<bool>, //false
    #[serde(default)]
    pub ignore_eos: Option<bool>, //false
    #[serde(default)]
    pub skip_special_tokens: Option<bool>, //false
    #[serde(default)]
    pub stop_token_ids: Option<Vec<usize>>, //[]
    #[serde(default)]
    pub logprobs: LogprobsField, // Accepts true/false boolean values only
    #[serde(default)]
    pub top_logprobs: TopLogprobsField, // Accepts boolean or number values
    pub repetition_penalty: Option<f32>, //1.1
    pub thinking: Option<bool>,       //false
    #[serde(default)]
    pub seed: Option<u64>, //random seed for reproducible sampling
}
