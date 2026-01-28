"use client";

import React from 'react';
import Monaco3DTrack from './Monaco3DTrack';

// This wrapper ensures Monaco3DTrack only runs on client
export default function Monaco3DTrackClient(props: any) {
    return <Monaco3DTrack {...props} />;
}
