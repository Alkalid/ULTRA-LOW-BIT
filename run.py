import llama_model

def run_main(checkpoint_path, temperature=1.0, topp=0.9, steps=256, 
             prompt=None, rng_seed=0, mode="generate", system_prompt=None, 
             tokenizer_path="tokenizer.bin"):
    # Set up configuration
    # Note: You might need to adapt these according to your actual C functions and structures
    prompt = "One day, Lily met a Shoggoth"

    # Build the transformer
    transformer = llama_model.Transformer()
    llama_model.build_transformer(transformer, checkpoint_path)

    llama_model.build_tokenizer(tokenizer_path, transformer.config.vocab_size)

    llama_model.build_sampler(transformer.config.vocab_size, temperature, topp, rng_seed)

    if mode == "generate" :
        llama_model.generate(transformer, prompt, steps)
    elif mode == "chat":
        # chat(&transformer, prompt, system_prompt, steps);
        return 0
    
# Example usage
run_main("stories42M.bin", prompt="Hello, world!")
