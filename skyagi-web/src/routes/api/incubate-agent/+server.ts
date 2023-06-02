import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import type { Config } from '@sveltejs/adapter-vercel';

// Can switch to the edge func if serverless is not necessary
export const config: Config = {
	runtime: 'nodejs18.x'
};

export const PUT = (async ({ request, locals }: { request: Request; locals: App.Locals }) => {
	const {
		conversation_id,
		agent_id,
		agent_name
	} = await request.json();

	// get existing agent
	const existing_agent = await locals.supabase
		.from('agent')
		.select('user_id, name, age, personality, initial_status')
		.eq('id', agent_id);
	
	// get memories
	const agent_memories = await locals.supabase
		.from('memory')
		.select('content')
		.eq('metadata->agent_id', agent_id)
		.eq('metadata->conversation_id', conversation_id);
	const memories = "";
	for (const mem of agent_memories) {
		memories += mem.content;
	}
	
	const res = await locals.supabase
		.from('agent')
		.insert({
			user_id: existing_agent[0].user_id,
			name: agent_name,
		    age: existing_agent[0].age,
			personality: existing_agent[0].personality,
			initial_status: existing_agent[0].status,
			initial_memory: memories
		});

	const { data } = await locals.supabase
		.from('agent')
		.select('id')
		.eq('user_id', existing_agent[0].user_id)
		.eq('name', agent_name)
		.eq('age', existing_agent[0].age)
		.eq('personality', existing_agent[0].personality)
		.eq('initial_status', existing_agent[0].status)
		.eq('initial_memory', memories);

	return new Response(JSON.stringify({ message: res, agent_id: data[0].id }), { status: 200 });

}) satisfies RequestHandler;