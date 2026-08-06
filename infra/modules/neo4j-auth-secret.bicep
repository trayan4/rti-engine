// Neo4j's official image expects NEO4J_AUTH as one combined
// "username/password" string, not a bare password. Composed here as its
// own module for the same reason postgres-dsn-secret.bicep exists: a
// nested resource's `parent` reference needs a plain string vault name,
// not a live cross-module expression.

param vaultName string

@secure()
param neo4jPassword string

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: vaultName
}

resource neo4jAuthSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'neo4j-auth'
  properties: {
    value: 'neo4j/${neo4jPassword}'
  }
}
