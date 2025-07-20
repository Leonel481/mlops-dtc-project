# Este código es compatible con Terraform 4.25.0 y versiones compatibles con 4.25.0.
# Para obtener información sobre la validación de este código de Terraform, consulta https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/google-cloud-platform-build#format-and-validate-the-configuration

#------------------------------------------------------------
#--------- Firewall Rules -----------------------------------
#------------------------------------------------------------

# Firewall HTTP
resource "google_compute_firewall" "allow_http" {
  name    = "allow-http"
  network = var.network  # Usa la variable para la red

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  target_tags   = ["http-server"]
  source_ranges = ["0.0.0.0/0"]
}

# Firewall HTTPS
resource "google_compute_firewall" "allow_https" {
  name    = "allow-https"
  network = var.network  # Usa la variable para la red

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  target_tags   = ["https-server"]
  source_ranges = ["0.0.0.0/0"]
}

#------------------------------------------------------------
#--------- Google Cloud Storage -----------------------------
#------------------------------------------------------------

resource "google_storage_bucket" "mlops_bucket" {
  name          = var.bucket_name
  location      = var.bucket_location
  force_destroy = true

  uniform_bucket_level_access = true
}

resource "google_storage_bucket_object" "data_raw_marker" {
  name   = "data_raw/.init"
  bucket = google_storage_bucket.mlops_bucket.name
  content = ""
}

resource "google_storage_bucket_object" "data_processed_marker" {
  name   = "data_processed/.init"
  bucket = google_storage_bucket.mlops_bucket.name
  content = ""
}

resource "google_storage_bucket_object" "models_marker" {
  name   = "models/.init"
  bucket = google_storage_bucket.mlops_bucket.name
  content = ""
}

resource "google_storage_bucket_object" "predictions_marker" {
  name   = "predictions/.init"
  bucket = google_storage_bucket.mlops_bucket.name
  content = ""
}

#------------------------------------------------------------
#--------- Compute Engine Instance --------------------------
#------------------------------------------------------------

resource "google_compute_instance" "vm-001-prod-scp-backend-uscentral" {
  boot_disk {
    auto_delete = true
    device_name = var.vm_name

    initialize_params {
      image = data.google_compute_image.ubuntu.self_link
      size  = var.boot_disk_size
      type  = "pd-balanced"
    }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  labels = {
    goog-ec-src   = "vm_add-tf"
    environment   = "prod"
    app           = "scp-backend"
  }

  machine_type = var.vm_machine_type
  name         = var.vm_name
  zone         = var.zone

  network_interface {
    access_config {
      nat_ip       = google_compute_address.static_external_ip.address
      network_tier = "PREMIUM"
    }

    network_ip  = google_compute_address.static_internal_ip.address
    queue_count = 0
    stack_type  = "IPV4_ONLY"
    subnetwork  = "projects/${var.gcp_project}/regions/${var.gcp_region}/subnetworks/default"  
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  service_account {
    email  = var.service_account
    scopes = ["https://www.googleapis.com/auth/devstorage.read_only", 
              "https://www.googleapis.com/auth/logging.write", 
              "https://www.googleapis.com/auth/monitoring.write", 
              "https://www.googleapis.com/auth/service.management.readonly", 
              "https://www.googleapis.com/auth/servicecontrol", 
              "https://www.googleapis.com/auth/trace.append"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  depends_on = [google_compute_address.static_external_ip, google_compute_address.static_internal_ip]


  tags = ["http-server", "https-server"]
  
}

#------------------------------------------------------------
#--------- Big Query ----------------------------------------
#------------------------------------------------------------

resource "google_bigquery_dataset" "mlops_dataset" {
  dataset_id                  = var.bq_dataset_id
  description                 = "Dataset for MLOps metrics"
  location                    = var.bq_location

}

resource "google_bigquery_table" "default" {
  dataset_id = google_bigquery_dataset.default.dataset_id
  table_id   = var.bq_table_name

  time_partitioning {
    type = "DAY"
  }

  labels = {
    env = "default"
  }

  schema = <<EOF
[
  {
    "name": "permalink",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "The Permalink"
  },
  {
    "name": "state",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "State where the head office is located"
  }
]
EOF

}


# Cloud Functions


# Cloud Run



