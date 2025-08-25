use candle_core::Tensor;
pub struct InputMetadata {
    pub prompt_lens: Vec<usize>,
    pub max_context_len: Option<usize>,
    pub block_tables: Option<Tensor>,
    pub context_lens: Option<Tensor>,
    pub slot_mapping: Tensor,
    pub is_prompt: bool,
}

impl InputMetadata {
    pub fn new(
        prompt_lens: Vec<usize>,
        max_context_len: Option<usize>,
        block_tables: Option<Tensor>,
        context_lens: Option<Tensor>,
        slot_mapping: Tensor,
    ) -> Self {
        let is_prompt = !prompt_lens.is_empty();
        Self {
            prompt_lens,
            max_context_len,
            block_tables,
            context_lens,
            slot_mapping,
            is_prompt,
        }
    }
}
