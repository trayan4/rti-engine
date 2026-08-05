// Blob storage for audit bundles.
//
// Every decision — the letter, the figures used, the review findings, the
// full trail — is written here as one JSON file per request. Independent
// of Postgres and the checkpoint store: a bundle should be retrievable
// even if the database that produced it is gone, because the compliance
// question it answers ("what did we send, and why") outlives the app.

@minLength(5)
param name string

param location string
param tags object

@description('Identity permitted to read and write bundles.')
param principalId string

// Built-in role: read, write and delete blob contents, not manage the
// account itself.
var storageBlobDataContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'

resource account 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'st${take(name, 22)}'
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    accessTier: 'Cool'
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: account
  name: 'default'
}

resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'audit-bundles'
  properties: {
    publicAccess: 'None'
  }
}

resource blobAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: account
  name: guid(account.id, principalId, storageBlobDataContributorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      storageBlobDataContributorRoleId
    )
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

output accountName string = account.name
output containerName string = container.name
output blobEndpoint string = account.properties.primaryEndpoints.blob
