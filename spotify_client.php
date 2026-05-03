<?php

function spotify_has_credentials(array $config): bool
{
    return ($config['client_id'] ?? '') !== 'YOUR_SPOTIFY_CLIENT_ID'
        && ($config['client_secret'] ?? '') !== 'YOUR_SPOTIFY_CLIENT_SECRET';
}

function spotify_get_access_token(array $config): ?string
{
    if (!spotify_has_credentials($config)) {
        return null;
    }

    $cacheFile = sys_get_temp_dir() . DIRECTORY_SEPARATOR . 'spotify_token_cache.json';
    if (file_exists($cacheFile)) {
        $cached = json_decode((string) file_get_contents($cacheFile), true);
        if (is_array($cached) && isset($cached['access_token'], $cached['expires_at']) && $cached['expires_at'] > time() + 30) {
            return $cached['access_token'];
        }
    }

    $ch = curl_init('https://accounts.spotify.com/api/token');
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => [
            'Authorization: Basic ' . base64_encode($config['client_id'] . ':' . $config['client_secret']),
            'Content-Type: application/x-www-form-urlencoded',
        ],
        CURLOPT_POSTFIELDS => http_build_query(['grant_type' => 'client_credentials']),
        CURLOPT_TIMEOUT => 15,
    ]);

    $response = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($response === false || $status !== 200) {
        return null;
    }

    $data = json_decode($response, true);
    if (!is_array($data) || !isset($data['access_token'], $data['expires_in'])) {
        return null;
    }

    $payload = [
        'access_token' => $data['access_token'],
        'expires_at' => time() + ((int) $data['expires_in']),
    ];
    file_put_contents($cacheFile, json_encode($payload));

    return $data['access_token'];
}

function spotify_search_tracks(string $token, string $query, string $market = 'TW', int $limit = 5): array
{
    $url = 'https://api.spotify.com/v1/search?' . http_build_query([
        'q' => $query,
        'type' => 'track',
        'market' => $market,
        'limit' => $limit,
    ]);

    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => ['Authorization: Bearer ' . $token],
        CURLOPT_TIMEOUT => 15,
    ]);

    $response = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($response === false || $status !== 200) {
        return [];
    }

    $data = json_decode($response, true);
    if (!is_array($data) || !isset($data['tracks']['items']) || !is_array($data['tracks']['items'])) {
        return [];
    }

    return $data['tracks']['items'];
}
