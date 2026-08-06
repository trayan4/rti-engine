// Key Vault, and the application's access to it.
//
// Secrets are passed in rather than generated here. A generated value
// would be written into the deployment history, which is readable by
// anyone with access to the subscription — so it would be a secret in
// name only.
//
// Access is granted by role assignment rather than an access policy.
// Policies are the older model and grant at vault scope with coarser
// verbs; RBAC grants read-only on secrets and nothing else.

param name string
param location string
param tags object

@description('The managed identity that will read these secrets.')
param principalId string

@secure()
param azureOpenAiApiKey string

@secure()
param anthropicApiKey string

@secure()
param langsmithApiKey string

@secure()
param groqApiKey string

@secure()
param pineconeApiKey string

@secure()
param neo4jPassword string

@secure()
param databaseAdminPassword string

// Built-in role: read secret contents, and nothing else. The application
// never lists, writes or deletes.
var secretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${take(name, 20)}'
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    // Recovery is on by default and cannot be disabled. Seven days is the
    // minimum, which matters when the same names are redeployed after a
    // teardown: a longer window would block recreating the vault.
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: null
    publicNetworkAccess: 'Enabled'
  }
}

var secrets = [
  { name: 'azure-openai-api-key', value: azureOpenAiApiKey }
  { name: 'anthropic-api-key', value: anthropicApiKey }
  { name: 'groq-api-key', value: groqApiKey }
  { name: 'pinecone-api-key', value: pineconeApiKey }
  { name: 'neo4j-password', value: neo4jPassword }
  { name: 'database-admin-password', value: databaseAdminPassword }
  { name: 'langsmith-api-key', value: langsmithApiKey }
]

resource storedSecrets 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = [
  for secret in secrets: {
    parent: vault
    name: secret.name
    properties: {
      value: secret.value
    }
  }
]

resource secretsAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: vault
  // A deterministic name, so redeploying updates the assignment rather
  // than failing on one that already exists.
  name: guid(vault.id, principalId, secretsUserRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      secretsUserRoleId
    )
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

output vaultName string = vault.name
output vaultUri string = vault.properties.vaultUri
