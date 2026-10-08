# Server migration plan
We will migrate the production database to the new cloud region over a weekend maintenance window. A rollback script restores the old cluster if replication lag exceeds the agreed limit.
