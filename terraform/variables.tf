variable "gcp_scp_key" {
    description = "GCP service account key file"
    type = string
}

variable "gcp_project" {
    description = "GCP project ID"
    type = string
}

variable "gcp_region" {
    description = "GCP Region"
    type = string
}

variable "service_account" {
    description = "Service account"
    type = string
}


variable "vm_name" {
    description = "instance name"
    type = string
}

variable "vm_machine_type" {
  description = "Machine type for the VM"
  type        = string
}

variable "boot_disk_size" {
  description = "Boot disk size in GB"
  type        = number
  default     = 20
}

variable "zone" {
  description = "zone configuration"
  type = string
}

variable "network" {
  description = "Network for the VM"
  type        = string
  default     = "default"
}

variable "bucket_name" {
  description = "Name of the Google Cloud Storage bucket"
  type        = string
}

variable "bucket_location" {
  description = "Location of the Google Cloud Storage bucket"
  type        = string
  default     = "US"
}

variable "bq_dataset_id" {
  description = "BigQuery dataset ID"
  type        = string
}

variable "bq_location" {
  description = "BigQuery dataset location"
  type        = string
  default     = "US"
}

variable "bq_table_name" {
  description = "BigQuery table name"
  type        = string
}