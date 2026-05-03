<?php
return [
    'client_id' => getenv('SPOTIFY_CLIENT_ID') ?: 'YOUR_SPOTIFY_CLIENT_ID',
    'client_secret' => getenv('SPOTIFY_CLIENT_SECRET') ?: 'YOUR_SPOTIFY_CLIENT_SECRET',
    'market' => 'TW',
    'results_limit' => 12,
];
