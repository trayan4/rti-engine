// Log Analytics and Application Insights.
//
// Created first because the container apps environment requires a
// workspace at creation and cannot be pointed at one afterwards.
//
// Application Insights is where the OpenTelemetry spans go once the
// system is deployed. Locally they go to Jaeger; the exporter is the only
// thing that differs.

param name string
param location string
param tags object

@description('How long telemetry is kept. Thirty days is the free floor.')
@minValue(30)
@maxValue(730)
param retentionInDays int = 30

resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${name}'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: retentionInDays
    features: {
      // Nothing here reads telemetry from outside the resource group.
      disableLocalAuth: false
    }
  }
}

resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${name}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: workspace.id
    // Ingestion is over the connection string held in Key Vault, so the
    // instrumentation key alone is not enough to write telemetry here.
    DisableLocalAuth: false
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

output workspaceId string = workspace.id
output workspaceCustomerId string = workspace.properties.customerId
output connectionString string = insights.properties.ConnectionString
