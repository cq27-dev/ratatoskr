//! Check the loader against a real installed plugin:
//! `cargo run -p ratatoskr-plugin --example probe -- <plugin-dir>`
#[tokio::main]
async fn main() {
    let dirs: Vec<std::path::PathBuf> = std::env::args().skip(1).map(Into::into).collect();
    let plugins = ratatoskr_plugin::discover(&dirs);
    for p in &plugins {
        println!(
            "plugin {} ({} hooks) at {}",
            p.name,
            p.hooks.len(),
            p.root.display()
        );
        for h in &p.hooks {
            println!(
                "  {} matcher={:?} timeout={:?}",
                h.event, h.matcher, h.timeout
            );
        }
        for s in &p.mcp_servers {
            println!("  mcpServer {} -> {:?} env={:?}", s.name, s.command, s.env);
        }
    }
    let cwd = std::env::current_dir().unwrap();
    let limits = ratatoskr_core::HookLimits::default();
    let contexts = ratatoskr_plugin::session_start(&plugins, &cwd, &limits).await;
    let names: Vec<String> = plugins.iter().map(|p| p.name.clone()).collect();
    match ratatoskr_plugin::compose(&contexts, &names, &limits) {
        Some(c) => println!("\n--- SessionStart context ({} chars) ---\n{c}", c.len()),
        None => println!("\n(no session context)"),
    }
}
