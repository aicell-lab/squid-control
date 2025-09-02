#!/usr/bin/env python3

import asyncio
import os

import pytest
from hypha_rpc import connect_to_server


@pytest.mark.asyncio
@pytest.mark.integration
async def test_connection():
    token = os.environ.get('AGENT_LENS_WORKSPACE_TOKEN')
    if not token:
        print('❌ No AGENT_LENS_WORKSPACE_TOKEN found in environment')
        return False

    print('🔗 Attempting to connect to Hypha server...')
    try:
        server = await connect_to_server({
            'server_url': 'https://hypha.aicell.io',
            'token': token,
            'workspace': 'agent-lens',
            'ping_interval': None
        })
        print('✅ Successfully connected to server')
        print(f'📊 Server workspace: {server.config.workspace}')
        return True
    except Exception as e:
        print(f'❌ Connection failed: {e}')
        return False

if __name__ == "__main__":
    result = asyncio.run(test_connection())
    exit(0 if result else 1)
