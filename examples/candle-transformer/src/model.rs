use std::sync::Arc;
use hibachi_core::Autoregressive;
use async_trait::async_trait;
use rand::{thread_rng, Rng};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};
use tokenizers::Tokenizer;
use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};
use tokio::sync::Mutex;

const MODEL_ID: &str = "HuggingFaceTB/SmolLM2-1.7B";
const EOS_TOKEN: &str = "<|endoftext|>";

pub struct Model {
    model: Llama,
    tokenizer: Tokenizer,
    logits: Mutex<LogitsProcessor>,
    cache: Mutex<model::Cache>
}

impl Model {
    pub fn new(temperature: Option<f64>, top_k: Option<usize>, top_p: Option<f64>) -> Self  {
        let dtype = DType::F16;
        let device = Device::Cpu;
        let api = Api::new().expect("creates api");
        let revision = "main".to_string();
        let api = api.repo(Repo::with_revision(MODEL_ID.to_string(), RepoType::Model, revision));
        let tokenizer_filename = api.get("tokenizer.json").expect("finds tokenizer");
        let config_filename = api.get("config.json").expect("Finds config");
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename).unwrap()).unwrap();
        let config = config.into_config(false);
        let filenames = vec![api.get("model.safetensors").unwrap()];
        let cache = model::Cache::new(false, dtype, &config, &device).unwrap();

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device).unwrap() };
        let llama = Llama::load(vb, &config).expect("Create llama");
        let tokenizer = Tokenizer::from_file(tokenizer_filename).expect("Create tokenizer");

        let mut logits_processor = {
            let temperature = temperature.unwrap_or(1.0);
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(299792458, sampling)
        };

        Self {
            model: llama,
            tokenizer: tokenizer,
            logits: Mutex::new(logits_processor),
            cache: Mutex::new(cache)
        }
    }

    pub fn stop_token(&self) -> Tensor {
        // hardcode since we know the stop token
        let device = Device::Cpu;
        Tensor::from_vec(vec![0u32],
                         &[1],
                         &device,
        ).expect("creates start token")
    }

    pub fn padding_token(&self) -> Tensor {
        // hardcode since we know the stop token
        let device = Device::Cpu;
        Tensor::from_vec(vec![1u32],
                         &[1],
                         &device,
        ).expect("creates start token")
    }

    pub fn tokenizer(&self) -> Tokenizer {
        self.tokenizer.clone()
    }
}

fn batchwise_logits(
    processor: &mut LogitsProcessor,
    logits: Tensor
) -> Tensor {

    let batch_size = logits.dims()[0];
    let vocab_size = logits.dims()[1];

    // Initialize a vector to hold the sampled tokens for each batch item
    let mut sampled_tokens = Vec::with_capacity(batch_size);

    // Process each batch item independently
    for batch_idx in 0..batch_size {
        // Extract logits for this batch item
        let batch_logits = logits.get(batch_idx).unwrap();

        // Use the existing LogitsProcessor to process this slice and get the token
        let token = processor.sample(&batch_logits).unwrap();
        sampled_tokens.push(token);
    }
    let out  = Tensor::from_slice(&sampled_tokens, &[batch_size], logits.device()).unwrap();
    out
}

#[async_trait]
impl Autoregressive for Model {
    type Sequence = Tensor;
    type Output = Tensor;

    async fn forward(&self, tensor: Self::Sequence) -> Self::Output {
        let mut cache = self.cache.lock().await;
        let sq_len = tensor.dims()[1];
        //println!("model forward");
        let logits = self.model.forward(&tensor, sq_len, &mut *cache).unwrap();
        let mut logits_processor = self.logits.lock().await;
        batchwise_logits(&mut *logits_processor, logits)
    }
}
