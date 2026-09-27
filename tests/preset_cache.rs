use lingxi_llm_client::{builtin_catalog, builtin_providers};

#[test]
fn catalog_is_shared_across_concurrent_callers() {
    let workers: Vec<_> = (0..16)
        .map(|_| {
            std::thread::spawn(|| {
                let catalog = builtin_catalog().expect("valid embedded catalog");
                (catalog.as_ptr() as usize, catalog.len())
            })
        })
        .collect();
    let catalog = builtin_catalog().unwrap();
    assert!(!catalog.is_empty());
    for worker in workers {
        assert_eq!(
            worker.join().unwrap(),
            (catalog.as_ptr() as usize, catalog.len())
        );
    }
}

#[test]
fn editable_builtin_profiles_do_not_modify_shared_catalog() {
    let catalog = builtin_catalog().unwrap();
    let mut editable = builtin_providers().unwrap();
    assert_eq!(editable, catalog);
    editable[0].base_url = "https://caller.example".into();
    editable[0].models.clear();
    assert_ne!(editable[0], catalog[0]);
    assert_eq!(builtin_providers().unwrap(), catalog);
}
