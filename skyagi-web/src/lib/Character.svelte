<script lang="ts">
	import type { CharacterType } from './types';
	import { Card, Avatar } from 'flowbite-svelte';

	export let character: CharacterType;
</script>

<Card>
	<div class="flex items-center space-x-4">
		<Avatar src={character.image} alt={character.image} class="flex-shrink-0" />
		<div class="flex-1 min-w-0">
			<p class="text-sm font-medium text-gray-900 truncate dark:text-white">
				{character.name}
			</p>
			<p class="text-sm text-gray-500 truncate dark:text-gray-400">
				{character.title}
			</p>
		</div>
	</div>
</Card>

<style>
</style>
