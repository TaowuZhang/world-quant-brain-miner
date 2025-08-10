-- Initialize Agent Network Database

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
    id VARCHAR PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    name VARCHAR NOT NULL,
    agent_type VARCHAR NOT NULL,
    status VARCHAR DEFAULT 'active',
    capabilities JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_heartbeat TIMESTAMP
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    id VARCHAR PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    sender_id VARCHAR REFERENCES agents(id),
    recipient_id VARCHAR REFERENCES agents(id),
    message_type VARCHAR NOT NULL,
    payload JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    status VARCHAR DEFAULT 'pending'
);

-- Create workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id VARCHAR PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    name VARCHAR NOT NULL,
    description TEXT,
    agent_id VARCHAR REFERENCES agents(id),
    workflow_type VARCHAR NOT NULL,
    configuration JSONB NOT NULL,
    status VARCHAR DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_executed TIMESTAMP,
    execution_count INTEGER DEFAULT 0
);

-- Create alpha_results table
CREATE TABLE IF NOT EXISTS alpha_results (
    id VARCHAR PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    expression TEXT NOT NULL,
    fitness VARCHAR,
    sharpe VARCHAR,
    turnover VARCHAR,
    returns VARCHAR,
    grade VARCHAR,
    checks JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_id VARCHAR REFERENCES agents(id)
);

-- Create system_events table
CREATE TABLE IF NOT EXISTS system_events (
    id VARCHAR PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    event_type VARCHAR NOT NULL,
    agent_id VARCHAR REFERENCES agents(id),
    payload JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_messages_recipient ON messages(recipient_id);
CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status);
CREATE INDEX IF NOT EXISTS idx_workflows_agent ON workflows(agent_id);
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_alpha_results_agent ON alpha_results(agent_id);
CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflows_updated_at BEFORE UPDATE ON workflows
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial data
INSERT INTO agents (id, name, agent_type, status, capabilities, metadata) VALUES
('system', 'System Agent', 'system', 'active', '{"system": true}', '{"description": "System management agent"}')
ON CONFLICT (id) DO NOTHING; 