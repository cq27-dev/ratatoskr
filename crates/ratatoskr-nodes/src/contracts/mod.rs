//! Typed contracts shared by the standard workflow and its Rust-owned operation hosts.

pub mod analyst;
pub mod context;
pub mod overseer;
pub mod scout;

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    #[test]
    fn legacy_module_paths_reexport_the_contract_types() {
        assert_eq!(
            TypeId::of::<crate::analyst::AnalystOutput>(),
            TypeId::of::<super::analyst::AnalystOutput>()
        );
        assert_eq!(
            TypeId::of::<crate::context::ContextOutput>(),
            TypeId::of::<super::context::ContextOutput>()
        );
        assert_eq!(
            TypeId::of::<crate::overseer::OverseerOutput>(),
            TypeId::of::<super::overseer::OverseerOutput>()
        );
        assert_eq!(
            TypeId::of::<crate::scout::ScoutOutput>(),
            TypeId::of::<super::scout::ScoutOutput>()
        );
    }
}
