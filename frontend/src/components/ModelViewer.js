import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { OrbitControls, Center, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

// Model component that loads and displays the OBJ file
function Model({ modelPath, bsdfParams }) {
    const [model, setModel] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const materialRef = useRef();

    const { scene } = useThree();

    useEffect(() => {
        if (!modelPath) return;

        setLoading(true);
        setError(null);

        const loader = new OBJLoader();

        // URL should point to the server endpoint
        const modelUrl = `http://localhost:3001/models/${modelPath.split('/').pop()}`;

        loader.load(
            modelUrl,
            (obj) => {
                // Create physical material and apply to all meshes
                const material = new THREE.MeshPhysicalMaterial({
                    color: '#cccccc',
                    roughness: 0.5,
                    metalness: 0.5,
                });
                materialRef.current = material;

                // Apply material to all meshes in the model
                obj.traverse((child) => {
                    if (child instanceof THREE.Mesh) {
                        child.material = material;
                    }
                });

                setModel(obj);
                setLoading(false);
            },
            (xhr) => {
                // Progress callback
                console.log((xhr.loaded / xhr.total) * 100 + '% loaded');
            },
            (err) => {
                console.error('Error loading model:', err);
                setError('Failed to load model');
                setLoading(false);
            }
        );
    }, [modelPath]);

    // Update material when BSDF parameters change
    useEffect(() => {
        if (materialRef.current && bsdfParams) {

            // Clearcoat
            if (bsdfParams.clearcoat) {
                materialRef.current.clearcoat = bsdfParams.clearcoat;
            }

            // Clearcoat gloss
            if (bsdfParams.clearcoat_gloss) {
                materialRef.current.clearcoatRoughness = 1.0 - bsdfParams.clearcoat_gloss;
            }

            // Metallic
            if (bsdfParams.metallic) {
                materialRef.current.metalness = bsdfParams.metallic;
            }

            // Specular
            if (bsdfParams.specular) {
                materialRef.current.specularIntensity = bsdfParams.specular;
            }

            // Roughness
            if (bsdfParams.roughness) {
                materialRef.current.roughness = bsdfParams.roughness;
            }

            // Base color
            if (bsdfParams.base_color) {
                materialRef.current.color = new THREE.Color().setRGB(
                    bsdfParams.base_color[0],
                    bsdfParams.base_color[1],
                    bsdfParams.base_color[2],
                    THREE.SRGBColorSpace
                );
            }

            // Anisotropy
            if (bsdfParams.anisotropic) {
                materialRef.current.anisotropy = bsdfParams.anisotropic;
            }

            // Specular tint
            if (bsdfParams.specular_tint) {
                // estimate specular color from specular tint
                // the color is a blend between the base color and white
                // based on the specular tint value
                const specularColor = new THREE.Color().copy(materialRef.current.color);
                specularColor.lerp(new THREE.Color(0xffffff), bsdfParams.specular_tint);
                materialRef.current.specular = specularColor;
            }

            // Sheen
            if (bsdfParams.sheen) {
                materialRef.current.sheen = bsdfParams.sheen;
            }

            // Sheen tint
            if (bsdfParams.sheen_tint) {
                // estimate sheen color from sheen tint
                // the color is a blend between the base color and white
                // based on the sheen tint value
                const sheenColor = new THREE.Color().copy(materialRef.current.color);
                sheenColor.lerp(new THREE.Color(0xffffff), bsdfParams.sheen_tint);
                materialRef.current.sheenTint = sheenColor;
            }

            // Force material update
            materialRef.current.needsUpdate = true;
        }
    }, [bsdfParams]);

    // Show a placeholder sphere if no model is loaded
    if (!model && !loading) {
        return (
            <mesh>
                <sphereGeometry args={[1, 16, 16]} />
                <meshStandardMaterial
                    ref={materialRef}
                />
            </mesh>
        );
    }

    return model ? <primitive object={model} scale={1.0} /> : null;
}

// Main ModelViewer component
const ModelViewer = ({ modelPath, bsdfParams }) => {
    return (
        <div style={{ width: '100%', height: '200px', borderRadius: '8px', overflow: 'hidden' }}>
            <Canvas>
                <PerspectiveCamera makeDefault position={[0, 0, 5]} />
                <ambientLight intensity={1} />
                <directionalLight position={[0, 5, 3]} intensity={1} />
                <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
                <Center>
                    <Model modelPath={modelPath} bsdfParams={bsdfParams} />
                </Center>
            </Canvas>
        </div>
    );
};

export default ModelViewer;
