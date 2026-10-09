import Testing
import HeteroMeshChecks

@Test func protocolConformance() throws { try runProtocolChecks() }

@Test func transformerReference() throws { try runTransformerChecks() }
@Test func transportPinPolicy() throws { try runTransportChecks() }

@Test func workerContractsAndLifecycle() throws { try runWorkerContractChecks() }
