// Infrastructure for the pay information request engine.
//
// Deployed into a resource group, so the group itself is the unit of
// teardown: deleting it removes everything this creates. That matters for
// a system that is deployed to be demonstrated and then removed.
//
// Secrets are passed in rather than generated here. A generated password
// would live in the deployment history, which is readable by anyone with
// access to the subscription.

targetScope = 'resourceGroup'

@description('Short name distinguishing this deployment. Used as a resource name prefix.')
@minLength(5)
@maxLength(12)
param name string = 'rtiengine'

@description('Where to deploy. Defaults to the resource group\'s region.')
param location string = resourceGroup().location

@description('Administrator login for the database.')
param databaseAdminUser string = 'rtiadmin'

@description('Administrator password for the database.')
@secure()
param databaseAdminPassword string

@description('Azure OpenAI endpoint for the models this system calls.')
  param azureOpenAiEndpoint string

@description('Deployment name for the reasoning model (e.g. gpt-5.6-terra).')
param azureOpenAiChatDeployment string

@description('Deployment name for the classification model (e.g. gpt-5.6-luna).')
param azureOpenAiMiniDeployment string

@description('Model name for the Anthropic fallback (e.g. claude-sonnet-5).')
param anthropicModel string

@description('Model name for the Groq fallback (e.g. llama-3.3-70b-versatile).')
param groqModel string

@description('Image tag to deploy. Only used when deployApps is true.')
param imageTag string = 'latest'

@secure()
param azureOpenAiApiKey string

@secure()
param anthropicApiKey string

@secure()
param groqApiKey string

@secure()
param pineconeApiKey string

param pineconeIndex string = 'rti-engine'

@description('Password for the graph database. Not an external service, but still a credential.')
@secure()
param neo4jPassword string

@description('Tags applied to everything, so a stray resource is traceable.')
param tags object = {
  application: 'rti-engine'
  managedBy: 'bicep'
}

// A suffix derived from the resource group makes names unique across
// subscriptions without anyone having to think of one, and keeps them
// stable across redeployments of the same group.
var suffix = uniqueString(resourceGroup().id)
var resourceName = '${name}${suffix}'

module observability 'modules/observability.bicep' = {
  name: 'observability'
  params: {
    name: resourceName
    location: location
    tags: tags
  }
}

module identity 'modules/identity.bicep' = {
  name: 'identity'
  params: {
    name: resourceName
    location: location
    tags: tags
  }
}

module secrets 'modules/secrets.bicep' = {
  name: 'secrets'
  params: {
    name: resourceName
    location: location
    tags: tags
    principalId: identity.outputs.principalId
    azureOpenAiApiKey: azureOpenAiApiKey
    anthropicApiKey: anthropicApiKey
    groqApiKey: groqApiKey
    pineconeApiKey: pineconeApiKey
    neo4jPassword: neo4jPassword
    databaseAdminPassword: databaseAdminPassword
  }
}

// The workspace's shared key is read here rather than returned from the
// observability module: a module output is recorded in the deployment
// history, and a key that grants log ingestion should not be.
resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: 'log-${resourceName}'
}

module platform 'modules/platform.bicep' = {
  name: 'platform'
  params: {
    name: resourceName
    location: location
    tags: tags
    workspaceCustomerId: observability.outputs.workspaceCustomerId
    workspaceSharedKey: workspace.listKeys().primarySharedKey
    principalId: identity.outputs.principalId
  }
}

module database 'modules/database.bicep' = {
  name: 'database'
  params: {
    name: resourceName
    location: 'eastus2'
    tags: tags
    administratorLogin: databaseAdminUser
    administratorPassword: databaseAdminPassword
  }
}

// The app reads one POSTGRES_DSN variable, not separate host/user/password
// fields, so the full connection string is assembled once here — with the
// password URL-encoded, since a raw '@' or similar breaks the URL just as
// it did in the migration workflow.
module postgresDsnSecret 'modules/postgres-dsn-secret.bicep' = {
  name: 'postgres-dsn-secret'
  params: {
    vaultName: secrets.outputs.vaultName
    databaseAdminUser: databaseAdminUser
    databaseAdminPassword: databaseAdminPassword
    databaseHost: database.outputs.host
    databaseName: database.outputs.databaseName
  }
}

module archive 'modules/archive.bicep' = {
  name: 'archive'
  params: {
    name: resourceName
    location: location
    tags: tags
    principalId: identity.outputs.principalId
  }
}

@description('False on the first deployment, before any image has been pushed. The registry and everything else still deploy; only the running apps wait.')
param deployApps bool = true

module apps 'modules/apps.bicep' = if (deployApps) {
  name: 'apps'
  params: {
    name: resourceName
    location: location
    tags: tags
    environmentId: platform.outputs.environmentId
    registryServer: platform.outputs.registryServer
    identityId: identity.outputs.identityId
    identityClientId: identity.outputs.clientId
    vaultUri: secrets.outputs.vaultUri
    azureOpenAiEndpoint: azureOpenAiEndpoint
    pineconeIndex: pineconeIndex
    appInsightsConnectionString: observability.outputs.connectionString
    azureOpenAiChatDeployment: azureOpenAiChatDeployment
    azureOpenAiMiniDeployment: azureOpenAiMiniDeployment
    anthropicModel: anthropicModel
    groqModel: groqModel
    imageTag: imageTag
  }
}

output apiUrl string = apps.?outputs.apiUrl ?? ''
output uiUrl string = apps.?outputs.uiUrl ?? ''
output vaultName string = secrets.outputs.vaultName
output identityId string = identity.outputs.identityId
output databaseHost string = database.outputs.host
output workspaceId string = observability.outputs.workspaceId
output appInsightsConnectionString string = observability.outputs.connectionString
output registryServer string = platform.outputs.registryServer
output environmentId string = platform.outputs.environmentId
