// The four running services.
//
// The MCP servers take internal ingress only. They hold the dataset and
// the authorization rules, and nothing outside this environment has any
// business calling them — the API reaches them over the environment's
// private DNS, and the internet cannot.
//
// This is also where the servers stop being subprocesses. Locally they
// are spawned over stdio; a pipe does not cross a container boundary, so
// here they are long-lived HTTP services.

@minLength(13)
param name string
param location string
param tags object

param environmentId string
param registryServer string
param identityId string

@description('Client ID of the identity, for Key Vault references.')
param identityClientId string

param vaultUri string
param azureOpenAiEndpoint string
param azureOpenAiChatDeployment string
param azureOpenAiMiniDeployment string
param azureOpenAiEmbeddingDeployment string
param anthropicModel string
param groqModel string
param pineconeIndex string
param appInsightsConnectionString string

@description('Image tag to deploy. Set by the release workflow.')
param imageTag string = 'latest'

var image = '${registryServer}/rti-engine:${imageTag}'

// Container app names have a 32-character limit. The full resourceName
// (prefix plus a 13-char unique suffix) is too long once combined with
// the longer prefixes like 'ca-mcp-analytics-'. Using only the unique
// suffix — always the last 13 characters — keeps every name under the
// limit while preserving the part that actually guarantees uniqueness.
var uniqueSuffix = substring(name, length(name) - 13, 13)

// Every app runs the same image with a different command, so one build
// serves all four and they cannot drift apart in dependencies.
var registries = [
  {
    server: registryServer
    identity: identityId
  }
]

var identityConfig = {
  type: 'UserAssigned'
  userAssignedIdentities: {
    '${identityId}': {}
  }
}

// A comprehension rather than a function: a Bicep function sees only its
// own arguments, so one referencing the vault would have to be handed the
// vault on every call.
var secretNames = [
  'azure-openai-api-key'
  'anthropic-api-key'
  'groq-api-key'
  'pinecone-api-key'
  'neo4j-password'
  'database-admin-password'
  'postgres-dsn'
]

var sharedSecrets = [
  for secretName in secretNames: {
    name: secretName
    keyVaultUrl: '${vaultUri}secrets/${secretName}'
    identity: identityId
  }
]

var neo4jSecret = {
  name: 'neo4j-password'
  keyVaultUrl: '${vaultUri}secrets/neo4j-password'
  identity: identityId
}

// The password is a secret reference rather than part of a connection
// string, because a full DSN in an environment variable would put the
// credential into every process listing and crash dump.
var sharedEnv = [
  { name: 'APP_ENV', value: 'azure' }
  { name: 'AZURE_CLIENT_ID', value: identityClientId }
  { name: 'AZURE_OPENAI_ENDPOINT', value: azureOpenAiEndpoint }
  { name: 'AZURE_OPENAI_API_KEY', secretRef: 'azure-openai-api-key' }
  { name: 'AZURE_OPENAI_CHAT_DEPLOYMENT', value: azureOpenAiChatDeployment }
  { name: 'AZURE_OPENAI_MINI_DEPLOYMENT', value: azureOpenAiMiniDeployment }
  { name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT', value: azureOpenAiEmbeddingDeployment }
  { name: 'ANTHROPIC_API_KEY', secretRef: 'anthropic-api-key' }
  { name: 'ANTHROPIC_MODEL', value: anthropicModel }
  { name: 'GROQ_API_KEY', secretRef: 'groq-api-key' }
  { name: 'GROQ_MODEL', value: groqModel }
  { name: 'PINECONE_API_KEY', secretRef: 'pinecone-api-key' }
  { name: 'PINECONE_INDEX', value: pineconeIndex }
  { name: 'NEO4J_PASSWORD', secretRef: 'neo4j-password' }
  { name: 'NEO4J_USERNAME', value: 'neo4j' }
  { name: 'NEO4J_URI', value: 'bolt://neo4j:7687' }
  { name: 'POSTGRES_DSN', secretRef: 'postgres-dsn' }
  {
    name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
    value: appInsightsConnectionString
  }
]

// The graph is rebuilt from source in seconds by a deterministic
// ingestion step, so persisting it would be cost and complexity for
// something regenerated on every deployment.
resource neo4j 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-neo4j-${uniqueSuffix}'
  location: location
  tags: tags
  identity: identityConfig
  properties: {
    environmentId: environmentId
    configuration: {
      secrets: [neo4jSecret]
      ingress: {
        external: false
        targetPort: 7687
        transport: 'tcp'
        exposedPort: 7687
      }
    }
    template: {
      containers: [
        {
          name: 'neo4j'
          image: 'neo4j:5-community'
          env: [
            { name: 'NEO4J_AUTH_FILE', value: '' }
            { name: 'NEO4J_PASSWORD', secretRef: 'neo4j-password' }
            {
              name: 'NEO4J_server_default__listen__address'
              value: '0.0.0.0'
            }
          ]
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
        }
      ]
      scale: {
        // A graph database that scaled to zero would lose its data on
        // every idle period, and scaling past one would give each replica
        // a different graph.
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

resource analyticsMcp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-mcp-analytics-${uniqueSuffix}'
  location: location
  tags: tags
  identity: identityConfig
  properties: {
    environmentId: environmentId
    configuration: {
      secrets: sharedSecrets
      registries: registries
      ingress: {
        // Internal only. This server holds the dataset and enforces the
        // tier rules; nothing outside the environment should reach it.
        external: false
        targetPort: 8080
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'analytics'
          image: image
          command: ['docker-entrypoint.sh', 'python', '-m', 'rti_engine.mcp.analytics_server']
          env: concat(sharedEnv, [
            { name: 'MCP_TRANSPORT', value: 'http' }
            { name: 'MCP_PORT', value: '8080' }
          ])
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

resource knowledgeMcp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-mcp-knowledge-${uniqueSuffix}'
  location: location
  tags: tags
  identity: identityConfig
  properties: {
    environmentId: environmentId
    configuration: {
      secrets: sharedSecrets
      registries: registries
      ingress: {
        external: false
        targetPort: 8080
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'knowledge'
          image: image
          command: ['docker-entrypoint.sh', 'python', '-m', 'rti_engine.mcp.knowledge_server']
          env: concat(sharedEnv, [
            { name: 'MCP_TRANSPORT', value: 'http' }
            { name: 'MCP_PORT', value: '8080' }
          ])
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-api-${uniqueSuffix}'
  location: location
  tags: tags
  identity: identityConfig
  properties: {
    environmentId: environmentId
    configuration: {
      secrets: sharedSecrets
      registries: registries
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          command: ['docker-entrypoint.sh', 'uvicorn', 'rti_engine.api.app:app', '--host', '0.0.0.0', '--port', '8000']
          env: concat(sharedEnv, [
            {
              name: 'ANALYTICS_MCP_URL'
              value: 'https://${analyticsMcp.properties.configuration.ingress.fqdn}/mcp'
            }
            {
              name: 'KNOWLEDGE_MCP_URL'
              value: 'https://${knowledgeMcp.properties.configuration.ingress.fqdn}/mcp'
            }
          ])
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
        }
      ]
      scale: {
        // A request in flight can be paused awaiting approval for days,
        // and its state is in Postgres rather than in memory — so a cold
        // start loses nothing, but a scale to zero mid-pipeline would.
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

resource ui 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-ui-${uniqueSuffix}'
  location: location
  tags: tags
  identity: identityConfig
  properties: {
    environmentId: environmentId
    configuration: {
      registries: registries
      ingress: {
        external: true
        targetPort: 8501
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'ui'
          image: image
          command: ['docker-entrypoint.sh', 'streamlit', 'run', 'src/rti_engine/ui/app.py', '--server.port', '8501', '--server.address', '0.0.0.0']
          env: [
            {
              name: 'RTI_API_URL'
              value: 'https://${api.properties.configuration.ingress.fqdn}'
            }
          ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output uiUrl string = 'https://${ui.properties.configuration.ingress.fqdn}'
