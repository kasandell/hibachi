use std::sync::Arc;
use hibachi::CandleForward;
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

const MODEL_NAME: &str = "SmoLM2-135M-Instruct";
const MODEL_ID: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
const EOS_TOKEN: &str = "</s>";

pub struct Model {
    model: Llama,
    tokenizer: Arc<Tokenizer>,
    logits: Mutex<LogitsProcessor>,
    cache: Mutex<model::Cache>
}

impl Model {
    pub fn new(top_k: Option<usize>, top_p: Option<f64>) -> Self  {
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
        // , tokenizer_filename, cache, config)

        let mut logits_processor = {
            let temperature = 1.0;
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
            tokenizer: Arc::new(tokenizer),
            logits: Mutex::new(logits_processor),
            cache: Mutex::new(cache)
        }
    }

    pub fn stop_token(&self) -> Tensor {
        let device = Device::Cpu;
        Tensor::from_vec(vec![0u32],
                         &[1],
                         &device,
        ).expect("creates start token")
            /*
        let val = self.tokenizer.token_to_id(EOS_TOKEN)
            .map(model::LlamaEosToks::Single);
        println!("{:?}", val);
        if let Some(model::LlamaEosToks::Single(data)) = val {

        } else {
            panic!("Unable to get eos token");
        }
             */

    }

    pub fn tokenizer(&self) -> Arc<Tokenizer> {
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

        // Add the token to our results
        sampled_tokens.push(token);
    }

    // Create a tensor from the sampled tokens
    Tensor::from_slice(&sampled_tokens, &[batch_size], logits.device()).unwrap()
}

#[async_trait]
impl CandleForward for Model {
    async fn forward(&self, tensor: Tensor) -> Tensor {
        let mut cache = self.cache.lock().await;
        let sq_len = tensor.dims()[1];
        print!(".");
        /*
        let (context_size, context_index) = if cache.use_kv_cache {
            (1, 0)
        } else {
            (sq_len, 0)
        };
         */
        //println!("Tensor: {:?}", tensor.dims());
        //println!("Sq len: {:?}", sq_len);

        let logits = self.model.forward(&tensor, sq_len, &mut *cache).unwrap();
        //println!("Did forward");
        //let logits = logits.squeeze(0).unwrap();
        let mut logits_processor = self.logits.lock().await;
        let out = batchwise_logits(&mut *logits_processor, logits);
        //println!("Out: {:?}", out.dims());
        out
        //let next_token = logits_processor.sample(&logits).unwrap();
    }
}
