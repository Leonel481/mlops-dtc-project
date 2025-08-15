# Este código es compatible con Terraform 4.25.0 y versiones compatibles con 4.25.0.
# Para obtener información sobre la validación de este código de Terraform, consulta https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/google-cloud-platform-build#format-and-validate-the-configuration

#------------------------------------------------------------
#--------- Firewall Rules -----------------------------------
#------------------------------------------------------------

# Firewall Prefect
resource "google_compute_firewall" "allow_prefect" {
  name    = "allow-prefect-ui"
  network = var.network  # Usa la variable para la red

  allow {
    protocol = "tcp"
    ports    = ["4200"]
  }

  target_tags   = ["prefect-ui"]
  source_ranges = ["0.0.0.0/0"]
}

# Firewall SSH IP
resource "google_compute_firewall" "allow-ssh_personal_ip" {
  name    = "allow-ssh-ip"
  network = var.network  # Usa la variable para la red

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  target_tags   = ["ssh-ip"]
  source_ranges = [var.ip]
}

#------------------------------------------------------------
#--------- Google Cloud Storage -----------------------------
#------------------------------------------------------------

resource "google_storage_bucket" "mlops-bucket" {
  name          = var.bucket_name
  location      = var.bucket_location
  force_destroy = true

  uniform_bucket_level_access = true

  labels = {
    project_type  = "vm-add-tf"
    environment   = "dev"
    app           = "ml-pipeline"
  }

}

# resource "google_storage_bucket_object" "data_raw_marker" {
#   name   = "data_raw/.init"
#   bucket = google_storage_bucket.mlops_bucket.name
#   content = ""
# }

# resource "google_storage_bucket_object" "data_processed_marker" {
#   name   = "data_processed/.init"
#   bucket = google_storage_bucket.mlops_bucket.name
#   content = ""
# }

# resource "google_storage_bucket_object" "artifacts_maker" {
#   name   = "artifacts/.init"
#   bucket = google_storage_bucket.mlops_bucket.name
#   content = ""
# }

# resource "google_storage_bucket_object" "models_maker" {
#   name   = "models/.init"
#   bucket = google_storage_bucket.mlops_bucket.name
#   content = ""
# }

# resource "google_storage_bucket_object" "metricss_marker" {
#   name   = "metrics/.init"
#   bucket = google_storage_bucket.mlops_bucket.name
#   content = ""
# }

#------------------------------------------------------------
#--------- Compute Engine Instance --------------------------
#------------------------------------------------------------

data "google_compute_image" "ubuntu" {
  family  = "ubuntu-2204-lts"
  project = "ubuntu-os-cloud"
}

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
    goog-ec-src   = "vm-add-tf"
    environment   = "dev"
    app           = "ml-pipeline"
  }

  machine_type = var.vm_machine_type
  name         = var.vm_name
  zone         = var.zone

  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    queue_count = 0
    stack_type  = "IPV4_ONLY"
    subnetwork  = "projects/${var.gcp_project}/regions/${var.gcp_region}/subnetworks/${var.network}"  
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  service_account {
    email  = var.service_account
    scopes = ["https://www.googleapis.com/auth/cloud-platform"] # Scope for all service in GCP
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  depends_on = []


  tags = ["prefect-ui", "ssh-ip"]
  
}

#------------------------------------------------------------
#--------- Google BigQuery Dataset --------------------------
#------------------------------------------------------------
resource "google_bigquery_dataset" "vertex_logging_dataset" {
  dataset_id                 = var.bq_dataset_id
  friendly_name              = "Vertex AI Metrics"
  description                = "Dataset to store prediction request and response logs from Vertex AI."
  location                   = var.bq_location
  delete_contents_on_destroy = true
}

resource "google_bigquery_table" "monitoring_metrics_table" {
  dataset_id = google_bigquery_dataset.vertex_logging_dataset.dataset_id
  table_id   = var.bq_table_name
  project    = var.gcp_project

  # Schema for table
  schema = jsonencode([
    {
      "name" = "timestamp"
      "type" = "TIMESTAMP"
      "mode" = "REQUIRED"
    },
    {
      "name" = "metric_type"
      "type" = "STRING"
      "mode" = "REQUIRED"
    },
    {
      "name" = "feature_name"
      "type" = "STRING"
      "mode" = "NULLABLE"
    },
    {
      "name" = "metric_value"
      "type" = "FLOAT"
      "mode" = "REQUIRED"
    }
  ])
}