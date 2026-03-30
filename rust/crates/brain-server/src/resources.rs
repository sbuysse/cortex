//! System resource monitoring with TTL caching.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use sysinfo::System;

/// How long to cache resource snapshots before refreshing.
const CACHE_TTL: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub cpu_percent: f32,
    pub memory_used_percent: f32,
    pub memory_available_gb: f32,
    pub disk_free_gb: f32,
    pub disk_used_percent: f32,
    pub ollama_running: bool,
}

pub struct ResourceManager {
    system: System,
    cached: Option<ResourceSnapshot>,
    last_refresh: Instant,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            system: System::new_all(),
            cached: None,
            last_refresh: Instant::now() - CACHE_TTL, // force first refresh
        }
    }

    pub fn get_snapshot(&mut self) -> ResourceSnapshot {
        // Return cached snapshot if within TTL
        if let Some(ref cached) = self.cached {
            if self.last_refresh.elapsed() < CACHE_TTL {
                return cached.clone();
            }
        }

        self.system.refresh_all();

        let cpu_percent = self.system.global_cpu_usage();
        let total_mem = self.system.total_memory() as f64;
        let used_mem = self.system.used_memory() as f64;
        let available_mem = (total_mem - used_mem) / (1024.0 * 1024.0 * 1024.0);
        let memory_used_percent = if total_mem > 0.0 {
            (used_mem / total_mem * 100.0) as f32
        } else {
            0.0
        };

        // Disk free space for root partition
        let disks = sysinfo::Disks::new_with_refreshed_list();
        let root_disk = disks
            .iter()
            .find(|d| d.mount_point() == std::path::Path::new("/"));
        let disk_free_gb = root_disk
            .map(|d| d.available_space() as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0) as f32;
        let disk_used_percent = root_disk
            .map(|d| {
                let total = d.total_space() as f64;
                let avail = d.available_space() as f64;
                if total > 0.0 { ((total - avail) / total * 100.0) as f32 } else { 0.0 }
            })
            .unwrap_or(0.0);

        // Check if Ollama is running
        let ollama_running = self
            .system
            .processes()
            .values()
            .any(|p| p.name().to_string_lossy().contains("ollama"));

        let snapshot = ResourceSnapshot {
            cpu_percent,
            memory_used_percent,
            memory_available_gb: available_mem as f32,
            disk_free_gb,
            disk_used_percent,
            ollama_running,
        };

        self.cached = Some(snapshot.clone());
        self.last_refresh = Instant::now();
        snapshot
    }
}
