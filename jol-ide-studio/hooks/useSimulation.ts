import { useState, useEffect } from 'react';
import { Flow, SystemChainNode, ProcessTreeNode } from '../types';

export interface SimulationState {
    flow: Flow | null;
    chain: SystemChainNode[];
    tree: ProcessTreeNode | null;
    resources: Record<string, any>;
    files: Record<string, string>;
    lastUpdate: string;
    isSimulating: boolean;
    refreshState: () => Promise<void>;
}

export const useSimulation = () => {
    const [state, setState] = useState<Omit<SimulationState, 'refreshState'>>({
        flow: null,
        chain: [],
        tree: null,
        resources: {},
        files: {},
        lastUpdate: '',
        isSimulating: false
    });

    const fetchState = async () => {
        try {
            const response = await fetch('/ide_state.json?t=' + Date.now());
            if (!response.ok) throw new Error('State file not found');

            const data = await response.json();

            setState({
                flow: data.flow || null,
                chain: data.chain || [],
                tree: data.tree || null,
                resources: data.resources || {},
                files: data.files || {},
                lastUpdate: new Date().toLocaleTimeString(),
                isSimulating: data.flow ? true : false
            });
        } catch (err) {
            console.debug("Simulation state load:", err);
        }
    };

    useEffect(() => {
        fetchState();
    }, []);

    return {
        ...state,
        refreshState: fetchState
    };
};
