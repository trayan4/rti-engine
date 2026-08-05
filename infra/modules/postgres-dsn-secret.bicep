// A separate module so the secret's `parent` reference is resolved inside
// its own deployment scope, where the vault name is an ordinary string
// parameter rather than a cross-module runtime expression. Bicep's
// static analysis of `parent`/`name` on a nested resource does not
// tolerate the latter.

param vaultName string
param databaseAdminUser string

@secure()
param databaseAdminPassword string

param databaseHost string
param databaseName string

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: vaultName
}

resource postgresDsnSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'postgres-dsn'
  properties: {
    value: 'postgresql://${databaseAdminUser}:${uriComponent(databaseAdminPassword)}@${databaseHost}:5432/${databaseName}?sslmode=require'
  }
}
