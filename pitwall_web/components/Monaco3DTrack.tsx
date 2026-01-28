"use client";

import React, { useEffect, useRef } from 'react';

interface Monaco3DTrackProps {
    d1Data: any[];
    d2Data: any[];
    hoverX: number | null;
}

export default function Monaco3DTrack({ d1Data, d2Data, hoverX }: Monaco3DTrackProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const sceneRef = useRef<any>(null);
    const rendererRef = useRef<any>(null);
    const cameraRef = useRef<any>(null);
    const animationRef = useRef<number>();
    const [isLoading, setIsLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);
    const hoverXRef = useRef<number | null>(null);

    useEffect(() => {
        if (typeof window === 'undefined' || !containerRef.current) return;

        // Prevent re-initialization if already set up
        if (rendererRef.current) {
            console.log('3D Track already initialized, skipping');
            return;
        }

        // Dynamically import Three.js only on client
        import('three').then((THREE) => {
            // Setup scene
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0c);
            sceneRef.current = scene;

            // Setup camera
            const camera = new THREE.PerspectiveCamera(
                50,
                containerRef.current!.clientWidth / containerRef.current!.clientHeight,
                0.1,
                1000
            );
            camera.position.set(0, 12, 12);
            camera.lookAt(0, 0, 0);
            cameraRef.current = camera;

            // Setup renderer
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(containerRef.current!.clientWidth, containerRef.current!.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            containerRef.current!.appendChild(renderer.domElement);
            rendererRef.current = renderer;

            // Add lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(10, 10, 5);
            scene.add(directionalLight);

            const pointLight = new THREE.PointLight(0x17c3ff, 0.5);
            pointLight.position.set(0, 5, 0);
            scene.add(pointLight);

            // Create track path
            const monacoPath = [
                [1.2, 0, 670.5], [0.0, 0, 613.3], [4.1, 0, 568.6], [10.4, 0, 533.4],
                [19.3, 0, 499.1], [28.1, 0, 476.3], [43.6, 0, 464.7], [71.2, 0, 459.8],
                [98.1, 0, 457.5], [138.0, 0, 451.5], [169.3, 0, 444.2], [208.1, 0, 434.5],
                [241.7, 0, 425.9], [283.1, 0, 419.2], [322.0, 0, 413.6], [374.7, 0, 389.4],
                [420.3, 0, 373.8], [475.1, 0, 365.9], [511.6, 0, 357.8], [549.7, 0, 333.0],
                [564.0, 0, 302.1], [564.2, 0, 273.5], [556.4, 0, 249.8], [534.9, 0, 218.6],
                [517.8, 0, 194.1], [516.1, 0, 170.0], [525.2, 0, 148.5], [538.7, 0, 129.3],
                [564.4, 0, 100.5], [585.2, 0, 71.8], [604.7, 0, 44.2], [624.1, 0, 20.5],
                [636.4, 0, 7.4], [655.7, 0, 1.0], [667.6, 0, 8.7], [673.2, 0, 19.6],
                [678.2, 0, 38.4], [685.4, 0, 59.4], [698.0, 0, 83.6], [707.7, 0, 94.0],
                [720.5, 0, 97.7], [725.6, 0, 90.8], [723.5, 0, 80.0], [721.3, 0, 75.6],
                [711.0, 0, 63.1], [701.2, 0, 52.7], [695.0, 0, 41.5], [695.5, 0, 29.9],
                [705.8, 0, 17.2], [721.9, 0, 8.9], [742.5, 0, 2.0], [759.7, 0, 0.0],
                [775.4, 0, 7.9], [779.4, 0, 22.0], [779.9, 0, 42.0], [778.5, 0, 62.8],
                [776.0, 0, 91.5], [774.5, 0, 117.4], [769.8, 0, 158.7], [759.1, 0, 197.6],
                [749.9, 0, 222.9], [727.3, 0, 274.2], [706.7, 0, 305.4], [663.6, 0, 340.0],
                [610.5, 0, 365.8], [583.2, 0, 379.7], [513.0, 0, 411.3], [485.8, 0, 420.4],
                [436.3, 0, 430.5], [404.1, 0, 434.1], [375.1, 0, 438.6], [363.5, 0, 449.9],
                [352.5, 0, 463.3], [325.1, 0, 463.0], [296.6, 0, 461.1], [276.6, 0, 464.2],
                [241.6, 0, 469.1], [209.0, 0, 473.0], [177.2, 0, 476.5], [134.1, 0, 479.9],
                [103.9, 0, 483.7], [77.8, 0, 505.4], [63.3, 0, 528.8], [54.1, 0, 555.0],
                [48.9, 0, 584.3], [48.0, 0, 623.8], [66.3, 0, 659.3], [82.2, 0, 697.3],
                [92.5, 0, 742.3], [96.6, 0, 778.3], [92.0, 0, 811.4], [81.6, 0, 825.3],
                [82.8, 0, 848.4], [91.1, 0, 876.4], [107.1, 0, 906.7], [126.0, 0, 929.2],
                [147.5, 0, 947.4], [161.3, 0, 957.2], [174.4, 0, 967.1], [182.5, 0, 979.4],
                [175.9, 0, 987.8], [164.4, 0, 994.4], [151.7, 0, 997.6], [133.8, 0, 1000.0],
                [109.2, 0, 998.8], [97.2, 0, 991.3], [92.3, 0, 980.1], [88.4, 0, 962.4],
                [76.4, 0, 935.9], [61.3, 0, 913.7], [45.6, 0, 882.7], [34.5, 0, 850.0],
                [17.2, 0, 788.2], [10.8, 0, 761.0], [4.2, 0, 713.2], [1.1, 0, 668.7]
            ];

            const points = monacoPath.map(p => new THREE.Vector3(
                (p[0] - 400) * 0.02,
                p[1],
                (p[2] - 500) * -0.02
            ));

            const curve = new THREE.CatmullRomCurve3(points, true);
            const trackPoints = curve.getPoints(200);

            // Create track line with colors
            const geometry = new THREE.BufferGeometry().setFromPoints(trackPoints);
            const colors = new Float32Array(trackPoints.length * 3);

            for (let i = 0; i < trackPoints.length; i++) {
                const dataIdx = Math.floor((i / trackPoints.length) * d1Data.length);
                const lecSpeed = d1Data[dataIdx]?.speed || 0;
                const saiSpeed = d2Data[dataIdx]?.speed || 0;
                const color = lecSpeed > saiSpeed ? new THREE.Color(0xdc2626) : new THREE.Color(0xfbbf24);
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            }

            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            const material = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 3 });
            const line = new THREE.Line(geometry, material);
            scene.add(line);

            // Add grid
            const gridHelper = new THREE.GridHelper(30, 30, 0x333333, 0x1a1a1e);
            gridHelper.position.y = -0.1;
            scene.add(gridHelper);

            // Create car markers (will be positioned based on hoverX)
            const carGeometry = new THREE.BoxGeometry(0.4, 0.2, 0.8);

            // LEC car (red)
            const lecMaterial = new THREE.MeshStandardMaterial({
                color: 0xdc2626,
                emissive: 0xdc2626,
                emissiveIntensity: 0.5,
                metalness: 0.8,
                roughness: 0.2
            });
            const lecCar = new THREE.Mesh(carGeometry, lecMaterial);
            lecCar.visible = false; // Hidden by default
            scene.add(lecCar);

            // SAI car (yellow)
            const saiMaterial = new THREE.MeshStandardMaterial({
                color: 0xfbbf24,
                emissive: 0xfbbf24,
                emissiveIntensity: 0.5,
                metalness: 0.8,
                roughness: 0.2
            });
            const saiCar = new THREE.Mesh(carGeometry, saiMaterial);
            saiCar.visible = false; // Hidden by default
            scene.add(saiCar);

            // Point lights for cars
            const lecLight = new THREE.PointLight(0xdc2626, 2, 3);
            lecCar.add(lecLight);
            const saiLight = new THREE.PointLight(0xfbbf24, 2, 3);
            saiCar.add(saiLight);

            // Add simple orbit controls
            let isDragging = false;
            let previousMousePosition = { x: 0, y: 0 };
            let cameraAngle = { theta: 0, phi: Math.PI / 4 };
            let cameraDistance = 15;

            const onMouseDown = (e: MouseEvent) => {
                isDragging = true;
                previousMousePosition = { x: e.clientX, y: e.clientY };
            };

            const onMouseMove = (e: MouseEvent) => {
                if (!isDragging) return;
                const deltaX = e.clientX - previousMousePosition.x;
                const deltaY = e.clientY - previousMousePosition.y;

                cameraAngle.theta += deltaX * 0.01;
                cameraAngle.phi = Math.max(0.1, Math.min(Math.PI / 2.2, cameraAngle.phi + deltaY * 0.01));

                previousMousePosition = { x: e.clientX, y: e.clientY };
            };

            const onMouseUp = () => {
                isDragging = false;
            };

            const onWheel = (e: WheelEvent) => {
                e.preventDefault();
                cameraDistance = Math.max(5, Math.min(25, cameraDistance + e.deltaY * 0.01));
            };

            renderer.domElement.addEventListener('mousedown', onMouseDown);
            renderer.domElement.addEventListener('mousemove', onMouseMove);
            renderer.domElement.addEventListener('mouseup', onMouseUp);
            renderer.domElement.addEventListener('wheel', onWheel);

            // Animation loop
            const animate = () => {
                animationRef.current = requestAnimationFrame(animate);

                // Update camera position
                camera.position.x = cameraDistance * Math.sin(cameraAngle.phi) * Math.cos(cameraAngle.theta);
                camera.position.y = cameraDistance * Math.cos(cameraAngle.phi);
                camera.position.z = cameraDistance * Math.sin(cameraAngle.phi) * Math.sin(cameraAngle.theta);
                camera.lookAt(0, 0, 0);

                // Update car positions based on hoverX
                if (hoverXRef.current !== null && d1Data.length > 0 && d2Data.length > 0) {
                    // Calculate position along track (0 to 1)
                    const progress = Math.max(0, Math.min(1, hoverXRef.current / 3400)); // 3400 is approx track length
                    const pointIndex = Math.floor(progress * trackPoints.length);

                    if (pointIndex < trackPoints.length) {
                        const trackPoint = trackPoints[pointIndex];

                        // Debug log (can remove later)
                        if (Math.random() < 0.01) { // Log occasionally to avoid spam
                            console.log('Car position update:', { hoverX: hoverXRef.current, progress, pointIndex });
                        }

                        // Position LEC car
                        lecCar.position.copy(trackPoint);
                        lecCar.position.y = 0.3; // Slightly above track
                        lecCar.visible = true;

                        // Position SAI car (slightly offset)
                        saiCar.position.copy(trackPoint);
                        saiCar.position.y = 0.3;
                        saiCar.position.x += 0.5; // Offset to the side
                        saiCar.visible = true;

                        // Calculate rotation to face forward along track
                        if (pointIndex < trackPoints.length - 1) {
                            const nextPoint = trackPoints[pointIndex + 1];
                            const direction = new THREE.Vector3().subVectors(nextPoint, trackPoint).normalize();
                            const angle = Math.atan2(direction.x, direction.z);
                            lecCar.rotation.y = angle;
                            saiCar.rotation.y = angle;
                        }
                    }
                } else {
                    // Hide cars when not hovering
                    lecCar.visible = false;
                    saiCar.visible = false;
                }

                renderer.render(scene, camera);
            };

            // Hide loading text and start animation
            setIsLoading(false);
            console.log('3D Track loaded successfully');
            animate();

            // Cleanup
            return () => {
                if (animationRef.current) {
                    cancelAnimationFrame(animationRef.current);
                }
                renderer.domElement.removeEventListener('mousedown', onMouseDown);
                renderer.domElement.removeEventListener('mousemove', onMouseMove);
                renderer.domElement.removeEventListener('mouseup', onMouseUp);
                renderer.domElement.removeEventListener('wheel', onWheel);
                renderer.dispose();
                containerRef.current?.removeChild(renderer.domElement);
            };
        }).catch((err: any) => {
            console.error('Failed to load Three.js:', err);
            setError('Failed to load 3D view: ' + err.message);
            setIsLoading(false);
        });
    }, [d1Data, d2Data]);

    // Update hoverX ref whenever prop changes
    useEffect(() => {
        hoverXRef.current = hoverX;
    }, [hoverX]);

    return (
        <div
            ref={containerRef}
            style={{
                width: '100%',
                height: '600px',
                background: '#0a0a0c',
                borderRadius: '16px',
                overflow: 'hidden',
                position: 'relative'
            }}
        >
            {isLoading && !error && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    color: '#17c3ff',
                    fontFamily: 'var(--font-oxanium)',
                    fontSize: '1.2rem',
                    fontWeight: 700,
                    zIndex: 10
                }}>
                    Loading 3D View...
                </div>
            )}
            {error && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    color: '#dc2626',
                    fontFamily: 'var(--font-oxanium)',
                    fontSize: '1rem',
                    fontWeight: 700,
                    zIndex: 10,
                    textAlign: 'center',
                    padding: '20px'
                }}>
                    {error}
                    <div style={{ fontSize: '0.8rem', marginTop: '10px', color: '#999' }}>
                        Check browser console for details
                    </div>
                </div>
            )}
        </div>
    );
}
