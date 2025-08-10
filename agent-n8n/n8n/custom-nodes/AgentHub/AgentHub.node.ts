import { IExecuteFunctions } from 'n8n-core';
import {
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';

export class AgentHub implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Agent Hub',
		name: 'agentHub',
		icon: 'file:agentHub.svg',
		group: ['transform'],
		version: 1,
		description: 'Interact with Agent Hub API',
		defaults: {
			name: 'Agent Hub',
		},
		inputs: ['main'],
		outputs: ['main'],
		properties: [
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				options: [
					{
						name: 'Generate Alpha',
						value: 'generateAlpha',
						description: 'Generate alpha factors',
					},
					{
						name: 'Test Alpha',
						value: 'testAlpha',
						description: 'Test an alpha expression',
					},
					{
						name: 'Get Agents',
						value: 'getAgents',
						description: 'Get list of agents',
					},
					{
						name: 'Send Message',
						value: 'sendMessage',
						description: 'Send message to agent',
					},
					{
						name: 'Execute Workflow',
						value: 'executeWorkflow',
						description: 'Execute a workflow',
					},
				],
				default: 'generateAlpha',
				noDataExpression: true,
			},
			{
				displayName: 'Agent Hub URL',
				name: 'agentHubUrl',
				type: 'string',
				default: 'http://agent-hub:8000',
				description: 'URL of the Agent Hub',
				displayOptions: {
					show: {
						operation: ['generateAlpha', 'testAlpha', 'getAgents', 'sendMessage', 'executeWorkflow'],
					},
				},
			},
			{
				displayName: 'Alpha Expression',
				name: 'alphaExpression',
				type: 'string',
				default: '',
				description: 'Alpha expression to test',
				displayOptions: {
					show: {
						operation: ['testAlpha'],
					},
				},
			},
			{
				displayName: 'Batch Size',
				name: 'batchSize',
				type: 'number',
				default: 5,
				description: 'Number of alphas to generate',
				displayOptions: {
					show: {
						operation: ['generateAlpha'],
					},
				},
			},
			{
				displayName: 'Sender ID',
				name: 'senderId',
				type: 'string',
				default: 'n8n',
				description: 'Sender agent ID',
				displayOptions: {
					show: {
						operation: ['sendMessage'],
					},
				},
			},
			{
				displayName: 'Recipient ID',
				name: 'recipientId',
				type: 'string',
				default: '',
				description: 'Recipient agent ID',
				displayOptions: {
					show: {
						operation: ['sendMessage'],
					},
				},
			},
			{
				displayName: 'Message Payload',
				name: 'messagePayload',
				type: 'json',
				default: '{}',
				description: 'Message payload',
				displayOptions: {
					show: {
						operation: ['sendMessage'],
					},
				},
			},
			{
				displayName: 'Workflow ID',
				name: 'workflowId',
				type: 'string',
				default: '',
				description: 'Workflow ID to execute',
				displayOptions: {
					show: {
						operation: ['executeWorkflow'],
					},
				},
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			try {
				const operation = this.getNodeParameter('operation', i) as string;
				const agentHubUrl = this.getNodeParameter('agentHubUrl', i) as string;

				let response: any;

				switch (operation) {
					case 'generateAlpha':
						const batchSize = this.getNodeParameter('batchSize', i) as number;
						response = await this.helpers.httpRequest({
							method: 'POST',
							url: `${agentHubUrl}/alpha/generate`,
							json: true,
							body: {
								batch_size: batchSize,
								parameters: {}
							},
						});
						break;

					case 'testAlpha':
						const alphaExpression = this.getNodeParameter('alphaExpression', i) as string;
						response = await this.helpers.httpRequest({
							method: 'POST',
							url: `${agentHubUrl}/alpha/generate`,
							json: true,
							body: {
								expression: alphaExpression
							},
						});
						break;

					case 'getAgents':
						response = await this.helpers.httpRequest({
							method: 'GET',
							url: `${agentHubUrl}/agents`,
							json: true,
						});
						break;

					case 'sendMessage':
						const senderId = this.getNodeParameter('senderId', i) as string;
						const recipientId = this.getNodeParameter('recipientId', i) as string;
						const messagePayload = this.getNodeParameter('messagePayload', i) as string;
						
						response = await this.helpers.httpRequest({
							method: 'POST',
							url: `${agentHubUrl}/messages`,
							json: true,
							body: {
								sender_id: senderId,
								recipient_id: recipientId,
								message_type: 'request',
								payload: JSON.parse(messagePayload),
								metadata: {}
							},
						});
						break;

					case 'executeWorkflow':
						const workflowId = this.getNodeParameter('workflowId', i) as string;
						response = await this.helpers.httpRequest({
							method: 'POST',
							url: `${agentHubUrl}/workflows/${workflowId}/execute`,
							json: true,
						});
						break;

					default:
						throw new Error(`Operation ${operation} not supported`);
				}

				returnData.push({
					json: response,
				});

			} catch (error) {
				if (this.continueOnFail()) {
					returnData.push({
						json: {
							error: error.message,
						},
					});
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}
} 