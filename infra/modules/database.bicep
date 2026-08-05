// PostgreSQL, holding both the application tables and the graph
// checkpoints.
//
// The smallest burstable tier. This is a system that is deployed to be
// demonstrated and then removed, and the workload is one request at a
// time — provisioning for throughput nobody will generate would be
// spending to look serious.

param name string
param location string
param tags object

param administratorLogin string

@secure()
param administratorPassword string

@description('Database the application connects to.')
param databaseName string = 'rti_engine'

@description('Postgres version. Matches what is run locally.')
param postgresVersion string = '15'

@description('Days of point-in-time restore. Seven is the minimum.')
@minValue(7)
@maxValue(35)
param backupRetentionDays int = 7

resource server 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: 'psql-${name}'
  location: location
  tags: tags
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: postgresVersion
    administratorLogin: administratorLogin
    administratorLoginPassword: administratorPassword
    storage: {
      storageSizeGB: 32
    }
    backup: {
      backupRetentionDays: backupRetentionDays
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
    network: {
      publicNetworkAccess: 'Enabled'
    }
  }
}

resource database 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: server
  name: databaseName
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

// Container apps reach the database from Azure's own address space. This
// rule admits that traffic and nothing from the public internet: the
// all-zero range is Azure's documented marker for internal services, not
// a wildcard.
resource allowAzureServices 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = {
  parent: server
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

output host string = server.properties.fullyQualifiedDomainName
output databaseName string = database.name
output serverName string = server.name
