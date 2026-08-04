// A managed identity for the running application.
//
// Created before anything it needs access to, because a role assignment
// names a principal and the principal has to exist first.
//
// User-assigned rather than system-assigned: several container apps share
// this identity, and a system-assigned one belongs to a single resource
// and disappears with it.

param name string
param location string
param tags object

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${name}'
  location: location
  tags: tags
}

output identityId string = identity.id
output principalId string = identity.properties.principalId
output clientId string = identity.properties.clientId
