// The container registry and the environment the apps run in.
//
// Separated from the apps themselves because these two outlive them: an
// app can be redeployed many times against the same registry and
// environment, and rebuilding either would mean re-pushing every image
// and reissuing every internal DNS name.

@description('Resource name prefix. Long enough that the registry name clears its own minimum.')
@minLength(5)
param name string

param location string
param tags object

@description('Log Analytics workspace the environment writes to.')
param workspaceCustomerId string

@secure()
@description('Shared key for that workspace. Required at creation and not readable afterwards.')
param workspaceSharedKey string

@description('Identity that pulls images from the registry.')
param principalId string

// Built-in role: pull images, and nothing else. The apps never push.
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'

resource registry 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  // Registry names permit no punctuation at all, so the prefix is folded in.
  name: 'acr${name}'
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    // Off deliberately: the apps authenticate with a managed identity, and
    // an admin user would be a shared password that every pull could use.
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

resource registryAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: registry
  name: guid(registry.id, principalId, acrPullRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      acrPullRoleId
    )
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${name}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspaceCustomerId
        sharedKey: workspaceSharedKey
      }
    }
    // Apps reach each other over the environment's internal DNS, which is
    // what lets the MCP servers stay unreachable from the internet while
    // remaining callable by the API.
    zoneRedundant: false
  }
}

output registryName string = registry.name
output registryServer string = registry.properties.loginServer
output environmentId string = environment.id
output environmentDomain string = environment.properties.defaultDomain
