/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
export const HASH_TABLE_MODEL_V2 = {
    modelTopology: {
        node: [
            {
                name: 'unknown_0',
                op: 'Const',
                attr: {
                    value: { tensor: { dtype: 'DT_INT32', tensorShape: {} } },
                    dtype: { type: 'DT_INT32' }
                }
            },
            {
                name: 'input',
                op: 'Placeholder',
                attr: { shape: { shape: { dim: [{ size: '-1' }] } }, dtype: { type: 'DT_STRING' } }
            },
            {
                name: 'unknown',
                op: 'Placeholder',
                attr: { shape: { shape: {} }, dtype: { type: 'DT_RESOURCE' } }
            },
            {
                name: 'StatefulPartitionedCall/None_Lookup/LookupTableFindV2',
                op: 'LookupTableFindV2',
                input: ['unknown', 'input', 'unknown_0'],
                attr: {
                    Tout: { type: 'DT_INT32' },
                    Tin: { type: 'DT_STRING' },
                    _has_manual_control_dependencies: { b: true }
                }
            },
            {
                name: 'Identity',
                op: 'Identity',
                input: ['StatefulPartitionedCall/None_Lookup/LookupTableFindV2'],
                attr: { T: { type: 'DT_INT32' } }
            }
        ],
        library: {},
        versions: { producer: 1240 }
    },
    format: 'graph-model',
    generatedBy: '2.11.0-dev20220822',
    convertedBy: 'TensorFlow.js Converter v1.7.0',
    weightSpecs: [
        { name: 'unknown_0', shape: [], dtype: 'int32' },
        { name: '114', shape: [2], dtype: 'string' },
        { name: '116', shape: [2], dtype: 'int32' }
    ],
    'weightData': new Uint8Array([
        0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x61, 0x01, 0x00,
        0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00
    ]).buffer,
    signature: {
        inputs: {
            input: {
                name: 'input:0',
                dtype: 'DT_STRING',
                tensorShape: { dim: [{ size: '-1' }] }
            },
            'unknown:0': {
                name: 'unknown:0',
                dtype: 'DT_RESOURCE',
                tensorShape: {},
                resourceId: 66
            }
        },
        outputs: {
            output_0: {
                name: 'Identity:0',
                dtype: 'DT_INT32',
                tensorShape: { dim: [{ size: '-1' }] }
            }
        }
    },
    modelInitializer: {
        node: [
            {
                name: 'Func/StatefulPartitionedCall/input_control_node/_0',
                op: 'NoOp',
                attr: { _has_manual_control_dependencies: { b: true } }
            },
            {
                name: '114',
                op: 'Const',
                attr: {
                    value: { tensor: { dtype: 'DT_STRING', tensorShape: { dim: [{ size: '2' }] } } },
                    _has_manual_control_dependencies: { b: true },
                    dtype: { type: 'DT_STRING' }
                }
            },
            {
                name: '116',
                op: 'Const',
                attr: {
                    _has_manual_control_dependencies: { b: true },
                    dtype: { type: 'DT_INT32' },
                    value: { tensor: { dtype: 'DT_INT32', tensorShape: { dim: [{ size: '2' }] } } }
                }
            },
            {
                name: 'Func/StatefulPartitionedCall/StatefulPartitionedCall/input_control_node/_9',
                op: 'NoOp',
                input: ['^Func/StatefulPartitionedCall/input_control_node/_0'],
                attr: { _has_manual_control_dependencies: { b: true } }
            },
            {
                name: 'StatefulPartitionedCall/StatefulPartitionedCall/hash_table',
                op: 'HashTableV2',
                input: [
                    '^Func/StatefulPartitionedCall/StatefulPartitionedCall/input_control_node/_9'
                ],
                attr: {
                    container: { s: '' },
                    use_node_name_sharing: { b: true },
                    _has_manual_control_dependencies: { b: true },
                    shared_name: { s: 'OTVfbG9hZF8xXzUy' },
                    value_dtype: { type: 'DT_INT32' },
                    key_dtype: { type: 'DT_STRING' }
                }
            },
            {
                name: 'Func/StatefulPartitionedCall/StatefulPartitionedCall/output_control_node/_11',
                op: 'NoOp',
                input: ['^StatefulPartitionedCall/StatefulPartitionedCall/hash_table'],
                attr: { _has_manual_control_dependencies: { b: true } }
            },
            {
                name: 'Func/StatefulPartitionedCall/output_control_node/_2',
                op: 'NoOp',
                input: [
                    '^Func/StatefulPartitionedCall/StatefulPartitionedCall/output_control_node/_11'
                ],
                attr: { _has_manual_control_dependencies: { b: true } }
            },
            {
                name: 'StatefulPartitionedCall/StatefulPartitionedCall/NoOp',
                op: 'NoOp',
                input: ['^StatefulPartitionedCall/StatefulPartitionedCall/hash_table'],
                attr: {
                    _acd_function_control_output: { b: true },
                    _has_manual_control_dependencies: { b: true }
                }
            },
            {
                name: 'StatefulPartitionedCall/StatefulPartitionedCall/Identity',
                op: 'Identity',
                input: [
                    'StatefulPartitionedCall/StatefulPartitionedCall/hash_table',
                    '^StatefulPartitionedCall/StatefulPartitionedCall/NoOp'
                ],
                attr: { T: { type: 'DT_RESOURCE' } }
            },
            {
                name: 'Func/StatefulPartitionedCall/StatefulPartitionedCall/output/_10',
                op: 'Identity',
                input: ['StatefulPartitionedCall/StatefulPartitionedCall/Identity'],
                attr: { T: { type: 'DT_RESOURCE' } }
            },
            {
                name: 'StatefulPartitionedCall/NoOp',
                op: 'NoOp',
                input: [
                    '^Func/StatefulPartitionedCall/StatefulPartitionedCall/output_control_node/_11'
                ],
                attr: {
                    _has_manual_control_dependencies: { b: true },
                    _acd_function_control_output: { b: true }
                }
            },
            {
                name: 'StatefulPartitionedCall/Identity',
                op: 'Identity',
                input: [
                    'Func/StatefulPartitionedCall/StatefulPartitionedCall/output/_10',
                    '^StatefulPartitionedCall/NoOp'
                ],
                attr: { T: { type: 'DT_RESOURCE' } }
            },
            {
                name: 'Func/StatefulPartitionedCall/output/_1',
                op: 'Identity',
                input: ['StatefulPartitionedCall/Identity'],
                attr: {
                    T: { type: 'DT_RESOURCE' },
                    _has_manual_control_dependencies: { b: true }
                }
            },
            {
                name: 'Func/StatefulPartitionedCall_1/input_control_node/_3',
                op: 'NoOp',
                input: ['^114', '^116', '^Func/StatefulPartitionedCall/output/_1'],
                attr: { _has_manual_control_dependencies: { b: true } }
            },
            {
                name: 'Func/StatefulPartitionedCall_1/input/_4',
                op: 'Identity',
                input: [
                    'Func/StatefulPartitionedCall/output/_1',
                    '^Func/StatefulPartitionedCall_1/input_control_node/_3'
                ],
                attr: { T: { type: 'DT_RESOURCE' } }
            },
            {
                name: 'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12',
                op: 'NoOp',
                input: ['^Func/StatefulPartitionedCall_1/input_control_node/_3'],
                attr: { _has_manual_control_dependencies: { b: true } }
            },
            {
                name: 'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_13',
                op: 'Identity',
                input: [
                    'Func/StatefulPartitionedCall_1/input/_4',
                    '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12'
                ],
                attr: { T: { type: 'DT_RESOURCE' } }
            },
            {
                name: 'Func/StatefulPartitionedCall_1/input/_5',
                op: 'Identity',
                input: ['114', '^Func/StatefulPartitionedCall_1/input_control_node/_3'],
                attr: { T: { type: 'DT_STRING' } }
            },
            {
                name: 'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_14',
                op: 'Identity',
                input: [
                    'Func/StatefulPartitionedCall_1/input/_5',
                    '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12'
                ],
                attr: { T: { type: 'DT_STRING' } }
            },
            {
                name: 'Func/StatefulPartitionedCall_1/input/_6',
                op: 'Identity',
                input: ['116', '^Func/StatefulPartitionedCall_1/input_control_node/_3'],
                attr: { T: { type: 'DT_INT32' } }
            },
            {
                name: 'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_15',
                op: 'Identity',
                input: [
                    'Func/StatefulPartitionedCall_1/input/_6',
                    '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12'
                ],
                attr: { T: { type: 'DT_INT32' } }
            },
            {
                name: 'StatefulPartitionedCall_1/StatefulPartitionedCall/key_value_init94/LookupTableImportV2',
                op: 'LookupTableImportV2',
                input: [
                    'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_13',
                    'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_14',
                    'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_15'
                ],
                attr: {
                    Tout: { type: 'DT_INT32' },
                    Tin: { type: 'DT_STRING' },
                    _has_manual_control_dependencies: { b: true }
                }
            },
            {
                name: 'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/output_control_node/_17',
                op: 'NoOp',
                input: [
                    '^StatefulPartitionedCall_1/StatefulPartitionedCall/key_value_init94/LookupTableImportV2'
                ],
                attr: { _has_manual_control_dependencies: { b: true } }
            },
            {
                name: 'Func/StatefulPartitionedCall_1/output_control_node/_8',
                op: 'NoOp',
                input: [
                    '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/output_control_node/_17'
                ],
                attr: { _has_manual_control_dependencies: { b: true } }
            },
            {
                name: 'NoOp',
                op: 'NoOp',
                input: [
                    '^Func/StatefulPartitionedCall/output_control_node/_2',
                    '^Func/StatefulPartitionedCall_1/output_control_node/_8'
                ],
                attr: {
                    _has_manual_control_dependencies: { b: true },
                    _acd_function_control_output: { b: true }
                }
            },
            {
                name: 'Identity',
                op: 'Identity',
                input: [
                    'Func/StatefulPartitionedCall/output/_1',
                    '^Func/StatefulPartitionedCall_1/output_control_node/_8', '^NoOp'
                ],
                attr: { T: { type: 'DT_RESOURCE' } }
            }
        ],
        versions: { producer: 1240 }
    },
    initializerSignature: {
        outputs: {
            'Identity:0': {
                name: 'Identity:0',
                dtype: 'DT_RESOURCE',
                tensorShape: {},
                resourceId: 66
            }
        }
    }
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaGFzaF90YWJsZV92Ml9tb2RlbF9sb2FkZXIuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvbnZlcnRlci9zcmMvZXhlY3V0b3IvdGVzdF9kYXRhL2hhc2hfdGFibGVfdjJfbW9kZWxfbG9hZGVyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE1BQU0sQ0FBQyxNQUFNLG1CQUFtQixHQUFHO0lBQ2pDLGFBQWEsRUFBRTtRQUNiLElBQUksRUFBRTtZQUNKO2dCQUNFLElBQUksRUFBRSxXQUFXO2dCQUNqQixFQUFFLEVBQUUsT0FBTztnQkFDWCxJQUFJLEVBQUU7b0JBQ0osS0FBSyxFQUFFLEVBQUMsTUFBTSxFQUFFLEVBQUMsS0FBSyxFQUFFLFVBQVUsRUFBRSxXQUFXLEVBQUUsRUFBRSxFQUFDLEVBQUM7b0JBQ3JELEtBQUssRUFBRSxFQUFDLElBQUksRUFBRSxVQUFVLEVBQUM7aUJBQzFCO2FBQ0Y7WUFDRDtnQkFDRSxJQUFJLEVBQUUsT0FBTztnQkFDYixFQUFFLEVBQUUsYUFBYTtnQkFDakIsSUFBSSxFQUNBLEVBQUMsS0FBSyxFQUFFLEVBQUMsS0FBSyxFQUFFLEVBQUMsR0FBRyxFQUFFLENBQUMsRUFBQyxJQUFJLEVBQUUsSUFBSSxFQUFDLENBQUMsRUFBQyxFQUFDLEVBQUUsS0FBSyxFQUFFLEVBQUMsSUFBSSxFQUFFLFdBQVcsRUFBQyxFQUFDO2FBQ3hFO1lBQ0Q7Z0JBQ0UsSUFBSSxFQUFFLFNBQVM7Z0JBQ2YsRUFBRSxFQUFFLGFBQWE7Z0JBQ2pCLElBQUksRUFBRSxFQUFDLEtBQUssRUFBRSxFQUFDLEtBQUssRUFBRSxFQUFFLEVBQUMsRUFBRSxLQUFLLEVBQUUsRUFBQyxJQUFJLEVBQUUsYUFBYSxFQUFDLEVBQUM7YUFDekQ7WUFDRDtnQkFDRSxJQUFJLEVBQUUsdURBQXVEO2dCQUM3RCxFQUFFLEVBQUUsbUJBQW1CO2dCQUN2QixL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LEVBQUUsT0FBTztpQkFDbEU7Z0JBQ0QsSUFBSSxFQUFFLEVBQUMsQ0FBQyxFQUFFLEVBQUMsSUFBSSxFQUFFLGFBQWEsRUFBQyxFQUFDO2FBQ2pDO1NBQ0Y7UUFDRCxRQUFRLEVBQUUsRUFBQyxRQUFRLEVBQUUsSUFBSSxFQUFDO0tBQzNCO0lBQ0Qsb0JBQW9CLEVBQUU7UUFDcEIsT0FBTyxFQUFFO1lBQ1AsWUFBWSxFQUFFO2dCQUNaLElBQUksRUFBRSxZQUFZO2dCQUNsQixLQUFLLEVBQUUsYUFBYTtnQkFDcEIsV0FBVyxFQUFFLEVBQUU7Z0JBQ2YsVUFBVSxFQUFFLEVBQUU7YUFDZjtTQUNGO0tBQ0Y7Q0FDRixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuZXhwb3J0IGNvbnN0IEhBU0hfVEFCTEVfTU9ERUxfVjIgPSB7XG4gIG1vZGVsVG9wb2xvZ3k6IHtcbiAgICBub2RlOiBbXG4gICAgICB7XG4gICAgICAgIG5hbWU6ICd1bmtub3duXzAnLFxuICAgICAgICBvcDogJ0NvbnN0JyxcbiAgICAgICAgYXR0cjoge1xuICAgICAgICAgIHZhbHVlOiB7dGVuc29yOiB7ZHR5cGU6ICdEVF9JTlQzMicsIHRlbnNvclNoYXBlOiB7fX19LFxuICAgICAgICAgIGR0eXBlOiB7dHlwZTogJ0RUX0lOVDMyJ31cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTogJ2lucHV0JyxcbiAgICAgICAgb3A6ICdQbGFjZWhvbGRlcicsXG4gICAgICAgIGF0dHI6XG4gICAgICAgICAgICB7c2hhcGU6IHtzaGFwZToge2RpbTogW3tzaXplOiAnLTEnfV19fSwgZHR5cGU6IHt0eXBlOiAnRFRfU1RSSU5HJ319XG4gICAgICB9LFxuICAgICAge1xuICAgICAgICBuYW1lOiAndW5rbm93bicsXG4gICAgICAgIG9wOiAnUGxhY2Vob2xkZXInLFxuICAgICAgICBhdHRyOiB7c2hhcGU6IHtzaGFwZToge319LCBkdHlwZToge3R5cGU6ICdEVF9SRVNPVVJDRSd9fVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTogJ1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL05vbmVfTG9va3VwL0xvb2t1cFRhYmxlRmluZFYyJyxcbiAgICAgICAgb3A6ICdMb29rdXBUYWJsZUZpbmRWMicsXG4gICAgICAgIGlucHV0OiBbJ3Vua25vd24nLCAnaW5wdXQnLCAndW5rbm93bl8wJ10sXG4gICAgICAgIGF0dHI6IHtcbiAgICAgICAgICBUb3V0OiB7dHlwZTogJ0RUX0lOVDMyJ30sXG4gICAgICAgICAgVGluOiB7dHlwZTogJ0RUX1NUUklORyd9LFxuICAgICAgICAgIF9oYXNfbWFudWFsX2NvbnRyb2xfZGVwZW5kZW5jaWVzOiB7YjogdHJ1ZX1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTogJ0lkZW50aXR5JyxcbiAgICAgICAgb3A6ICdJZGVudGl0eScsXG4gICAgICAgIGlucHV0OiBbJ1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL05vbmVfTG9va3VwL0xvb2t1cFRhYmxlRmluZFYyJ10sXG4gICAgICAgIGF0dHI6IHtUOiB7dHlwZTogJ0RUX0lOVDMyJ319XG4gICAgICB9XG4gICAgXSxcbiAgICBsaWJyYXJ5OiB7fSxcbiAgICB2ZXJzaW9uczoge3Byb2R1Y2VyOiAxMjQwfVxuICB9LFxuICBmb3JtYXQ6ICdncmFwaC1tb2RlbCcsXG4gIGdlbmVyYXRlZEJ5OiAnMi4xMS4wLWRldjIwMjIwODIyJyxcbiAgY29udmVydGVkQnk6ICdUZW5zb3JGbG93LmpzIENvbnZlcnRlciB2MS43LjAnLFxuICB3ZWlnaHRTcGVjczogW1xuICAgIHtuYW1lOiAndW5rbm93bl8wJywgc2hhcGU6IFtdLCBkdHlwZTogJ2ludDMyJ30sXG4gICAge25hbWU6ICcxMTQnLCBzaGFwZTogWzJdLCBkdHlwZTogJ3N0cmluZyd9LFxuICAgIHtuYW1lOiAnMTE2Jywgc2hhcGU6IFsyXSwgZHR5cGU6ICdpbnQzMid9XG4gIF0sXG4gICd3ZWlnaHREYXRhJzpcbiAgICAgIG5ldyBVaW50OEFycmF5KFtcbiAgICAgICAgMHhmZiwgMHhmZiwgMHhmZiwgMHhmZiwgMHgwMSwgMHgwMCwgMHgwMCwgMHgwMCwgMHg2MSwgMHgwMSwgMHgwMCxcbiAgICAgICAgMHgwMCwgMHgwMCwgMHg2MiwgMHgwMCwgMHgwMCwgMHgwMCwgMHgwMCwgMHgwMSwgMHgwMCwgMHgwMCwgMHgwMFxuICAgICAgXSkuYnVmZmVyLFxuXG4gIHNpZ25hdHVyZToge1xuICAgIGlucHV0czoge1xuICAgICAgaW5wdXQ6IHtcbiAgICAgICAgbmFtZTogJ2lucHV0OjAnLFxuICAgICAgICBkdHlwZTogJ0RUX1NUUklORycsXG4gICAgICAgIHRlbnNvclNoYXBlOiB7ZGltOiBbe3NpemU6ICctMSd9XX1cbiAgICAgIH0sXG4gICAgICAndW5rbm93bjowJzoge1xuICAgICAgICBuYW1lOiAndW5rbm93bjowJyxcbiAgICAgICAgZHR5cGU6ICdEVF9SRVNPVVJDRScsXG4gICAgICAgIHRlbnNvclNoYXBlOiB7fSxcbiAgICAgICAgcmVzb3VyY2VJZDogNjZcbiAgICAgIH1cbiAgICB9LFxuICAgIG91dHB1dHM6IHtcbiAgICAgIG91dHB1dF8wOiB7XG4gICAgICAgIG5hbWU6ICdJZGVudGl0eTowJyxcbiAgICAgICAgZHR5cGU6ICdEVF9JTlQzMicsXG4gICAgICAgIHRlbnNvclNoYXBlOiB7ZGltOiBbe3NpemU6ICctMSd9XX1cbiAgICAgIH1cbiAgICB9XG4gIH0sXG4gIG1vZGVsSW5pdGlhbGl6ZXI6IHtcbiAgICBub2RlOiBbXG4gICAgICB7XG4gICAgICAgIG5hbWU6ICdGdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2lucHV0X2NvbnRyb2xfbm9kZS9fMCcsXG4gICAgICAgIG9wOiAnTm9PcCcsXG4gICAgICAgIGF0dHI6IHtfaGFzX21hbnVhbF9jb250cm9sX2RlcGVuZGVuY2llczoge2I6IHRydWV9fVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTogJzExNCcsXG4gICAgICAgIG9wOiAnQ29uc3QnLFxuICAgICAgICBhdHRyOiB7XG4gICAgICAgICAgdmFsdWU6XG4gICAgICAgICAgICAgIHt0ZW5zb3I6IHtkdHlwZTogJ0RUX1NUUklORycsIHRlbnNvclNoYXBlOiB7ZGltOiBbe3NpemU6ICcyJ31dfX19LFxuICAgICAgICAgIF9oYXNfbWFudWFsX2NvbnRyb2xfZGVwZW5kZW5jaWVzOiB7YjogdHJ1ZX0sXG4gICAgICAgICAgZHR5cGU6IHt0eXBlOiAnRFRfU1RSSU5HJ31cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTogJzExNicsXG4gICAgICAgIG9wOiAnQ29uc3QnLFxuICAgICAgICBhdHRyOiB7XG4gICAgICAgICAgX2hhc19tYW51YWxfY29udHJvbF9kZXBlbmRlbmNpZXM6IHtiOiB0cnVlfSxcbiAgICAgICAgICBkdHlwZToge3R5cGU6ICdEVF9JTlQzMid9LFxuICAgICAgICAgIHZhbHVlOlxuICAgICAgICAgICAgICB7dGVuc29yOiB7ZHR5cGU6ICdEVF9JTlQzMicsIHRlbnNvclNoYXBlOiB7ZGltOiBbe3NpemU6ICcyJ31dfX19XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6XG4gICAgICAgICAgICAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9pbnB1dF9jb250cm9sX25vZGUvXzknLFxuICAgICAgICBvcDogJ05vT3AnLFxuICAgICAgICBpbnB1dDogWydeRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9pbnB1dF9jb250cm9sX25vZGUvXzAnXSxcbiAgICAgICAgYXR0cjoge19oYXNfbWFudWFsX2NvbnRyb2xfZGVwZW5kZW5jaWVzOiB7YjogdHJ1ZX19XG4gICAgICB9LFxuICAgICAge1xuICAgICAgICBuYW1lOiAnU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvaGFzaF90YWJsZScsXG4gICAgICAgIG9wOiAnSGFzaFRhYmxlVjInLFxuICAgICAgICBpbnB1dDogW1xuICAgICAgICAgICdeRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9pbnB1dF9jb250cm9sX25vZGUvXzknXG4gICAgICAgIF0sXG4gICAgICAgIGF0dHI6IHtcbiAgICAgICAgICBjb250YWluZXI6IHtzOiAnJ30sXG4gICAgICAgICAgdXNlX25vZGVfbmFtZV9zaGFyaW5nOiB7YjogdHJ1ZX0sXG4gICAgICAgICAgX2hhc19tYW51YWxfY29udHJvbF9kZXBlbmRlbmNpZXM6IHtiOiB0cnVlfSxcbiAgICAgICAgICBzaGFyZWRfbmFtZToge3M6ICdPVFZmYkc5aFpGOHhYelV5J30sXG4gICAgICAgICAgdmFsdWVfZHR5cGU6IHt0eXBlOiAnRFRfSU5UMzInfSxcbiAgICAgICAgICBrZXlfZHR5cGU6IHt0eXBlOiAnRFRfU1RSSU5HJ31cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTpcbiAgICAgICAgICAgICdGdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL291dHB1dF9jb250cm9sX25vZGUvXzExJyxcbiAgICAgICAgb3A6ICdOb09wJyxcbiAgICAgICAgaW5wdXQ6IFsnXlN0YXRlZnVsUGFydGl0aW9uZWRDYWxsL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2hhc2hfdGFibGUnXSxcbiAgICAgICAgYXR0cjoge19oYXNfbWFudWFsX2NvbnRyb2xfZGVwZW5kZW5jaWVzOiB7YjogdHJ1ZX19XG4gICAgICB9LFxuICAgICAge1xuICAgICAgICBuYW1lOiAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9vdXRwdXRfY29udHJvbF9ub2RlL18yJyxcbiAgICAgICAgb3A6ICdOb09wJyxcbiAgICAgICAgaW5wdXQ6IFtcbiAgICAgICAgICAnXkZ1bmMvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvb3V0cHV0X2NvbnRyb2xfbm9kZS9fMTEnXG4gICAgICAgIF0sXG4gICAgICAgIGF0dHI6IHtfaGFzX21hbnVhbF9jb250cm9sX2RlcGVuZGVuY2llczoge2I6IHRydWV9fVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTogJ1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL05vT3AnLFxuICAgICAgICBvcDogJ05vT3AnLFxuICAgICAgICBpbnB1dDogWydeU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvaGFzaF90YWJsZSddLFxuICAgICAgICBhdHRyOiB7XG4gICAgICAgICAgX2FjZF9mdW5jdGlvbl9jb250cm9sX291dHB1dDoge2I6IHRydWV9LFxuICAgICAgICAgIF9oYXNfbWFudWFsX2NvbnRyb2xfZGVwZW5kZW5jaWVzOiB7YjogdHJ1ZX1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTogJ1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL0lkZW50aXR5JyxcbiAgICAgICAgb3A6ICdJZGVudGl0eScsXG4gICAgICAgIGlucHV0OiBbXG4gICAgICAgICAgJ1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2hhc2hfdGFibGUnLFxuICAgICAgICAgICdeU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvTm9PcCdcbiAgICAgICAgXSxcbiAgICAgICAgYXR0cjoge1Q6IHt0eXBlOiAnRFRfUkVTT1VSQ0UnfX1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6ICdGdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL291dHB1dC9fMTAnLFxuICAgICAgICBvcDogJ0lkZW50aXR5JyxcbiAgICAgICAgaW5wdXQ6IFsnU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvSWRlbnRpdHknXSxcbiAgICAgICAgYXR0cjoge1Q6IHt0eXBlOiAnRFRfUkVTT1VSQ0UnfX1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6ICdTdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9Ob09wJyxcbiAgICAgICAgb3A6ICdOb09wJyxcbiAgICAgICAgaW5wdXQ6IFtcbiAgICAgICAgICAnXkZ1bmMvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvb3V0cHV0X2NvbnRyb2xfbm9kZS9fMTEnXG4gICAgICAgIF0sXG4gICAgICAgIGF0dHI6IHtcbiAgICAgICAgICBfaGFzX21hbnVhbF9jb250cm9sX2RlcGVuZGVuY2llczoge2I6IHRydWV9LFxuICAgICAgICAgIF9hY2RfZnVuY3Rpb25fY29udHJvbF9vdXRwdXQ6IHtiOiB0cnVlfVxuICAgICAgICB9XG4gICAgICB9LFxuICAgICAge1xuICAgICAgICBuYW1lOiAnU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvSWRlbnRpdHknLFxuICAgICAgICBvcDogJ0lkZW50aXR5JyxcbiAgICAgICAgaW5wdXQ6IFtcbiAgICAgICAgICAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9vdXRwdXQvXzEwJyxcbiAgICAgICAgICAnXlN0YXRlZnVsUGFydGl0aW9uZWRDYWxsL05vT3AnXG4gICAgICAgIF0sXG4gICAgICAgIGF0dHI6IHtUOiB7dHlwZTogJ0RUX1JFU09VUkNFJ319XG4gICAgICB9LFxuICAgICAge1xuICAgICAgICBuYW1lOiAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9vdXRwdXQvXzEnLFxuICAgICAgICBvcDogJ0lkZW50aXR5JyxcbiAgICAgICAgaW5wdXQ6IFsnU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvSWRlbnRpdHknXSxcbiAgICAgICAgYXR0cjoge1xuICAgICAgICAgIFQ6IHt0eXBlOiAnRFRfUkVTT1VSQ0UnfSxcbiAgICAgICAgICBfaGFzX21hbnVhbF9jb250cm9sX2RlcGVuZGVuY2llczoge2I6IHRydWV9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6ICdGdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsXzEvaW5wdXRfY29udHJvbF9ub2RlL18zJyxcbiAgICAgICAgb3A6ICdOb09wJyxcbiAgICAgICAgaW5wdXQ6IFsnXjExNCcsICdeMTE2JywgJ15GdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL291dHB1dC9fMSddLFxuICAgICAgICBhdHRyOiB7X2hhc19tYW51YWxfY29udHJvbF9kZXBlbmRlbmNpZXM6IHtiOiB0cnVlfX1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6ICdGdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsXzEvaW5wdXQvXzQnLFxuICAgICAgICBvcDogJ0lkZW50aXR5JyxcbiAgICAgICAgaW5wdXQ6IFtcbiAgICAgICAgICAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9vdXRwdXQvXzEnLFxuICAgICAgICAgICdeRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL2lucHV0X2NvbnRyb2xfbm9kZS9fMydcbiAgICAgICAgXSxcbiAgICAgICAgYXR0cjoge1Q6IHt0eXBlOiAnRFRfUkVTT1VSQ0UnfX1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6XG4gICAgICAgICAgICAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2lucHV0X2NvbnRyb2xfbm9kZS9fMTInLFxuICAgICAgICBvcDogJ05vT3AnLFxuICAgICAgICBpbnB1dDogWydeRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL2lucHV0X2NvbnRyb2xfbm9kZS9fMyddLFxuICAgICAgICBhdHRyOiB7X2hhc19tYW51YWxfY29udHJvbF9kZXBlbmRlbmNpZXM6IHtiOiB0cnVlfX1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6XG4gICAgICAgICAgICAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2lucHV0L18xMycsXG4gICAgICAgIG9wOiAnSWRlbnRpdHknLFxuICAgICAgICBpbnB1dDogW1xuICAgICAgICAgICdGdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsXzEvaW5wdXQvXzQnLFxuICAgICAgICAgICdeRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2lucHV0X2NvbnRyb2xfbm9kZS9fMTInXG4gICAgICAgIF0sXG4gICAgICAgIGF0dHI6IHtUOiB7dHlwZTogJ0RUX1JFU09VUkNFJ319XG4gICAgICB9LFxuICAgICAge1xuICAgICAgICBuYW1lOiAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL2lucHV0L181JyxcbiAgICAgICAgb3A6ICdJZGVudGl0eScsXG4gICAgICAgIGlucHV0OiBbJzExNCcsICdeRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL2lucHV0X2NvbnRyb2xfbm9kZS9fMyddLFxuICAgICAgICBhdHRyOiB7VDoge3R5cGU6ICdEVF9TVFJJTkcnfX1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6XG4gICAgICAgICAgICAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2lucHV0L18xNCcsXG4gICAgICAgIG9wOiAnSWRlbnRpdHknLFxuICAgICAgICBpbnB1dDogW1xuICAgICAgICAgICdGdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsXzEvaW5wdXQvXzUnLFxuICAgICAgICAgICdeRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2lucHV0X2NvbnRyb2xfbm9kZS9fMTInXG4gICAgICAgIF0sXG4gICAgICAgIGF0dHI6IHtUOiB7dHlwZTogJ0RUX1NUUklORyd9fVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTogJ0Z1bmMvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGxfMS9pbnB1dC9fNicsXG4gICAgICAgIG9wOiAnSWRlbnRpdHknLFxuICAgICAgICBpbnB1dDogWycxMTYnLCAnXkZ1bmMvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGxfMS9pbnB1dF9jb250cm9sX25vZGUvXzMnXSxcbiAgICAgICAgYXR0cjoge1Q6IHt0eXBlOiAnRFRfSU5UMzInfX1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6XG4gICAgICAgICAgICAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2lucHV0L18xNScsXG4gICAgICAgIG9wOiAnSWRlbnRpdHknLFxuICAgICAgICBpbnB1dDogW1xuICAgICAgICAgICdGdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsXzEvaW5wdXQvXzYnLFxuICAgICAgICAgICdeRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2lucHV0X2NvbnRyb2xfbm9kZS9fMTInXG4gICAgICAgIF0sXG4gICAgICAgIGF0dHI6IHtUOiB7dHlwZTogJ0RUX0lOVDMyJ319XG4gICAgICB9LFxuICAgICAge1xuICAgICAgICBuYW1lOlxuICAgICAgICAgICAgJ1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsXzEvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwva2V5X3ZhbHVlX2luaXQ5NC9Mb29rdXBUYWJsZUltcG9ydFYyJyxcbiAgICAgICAgb3A6ICdMb29rdXBUYWJsZUltcG9ydFYyJyxcbiAgICAgICAgaW5wdXQ6IFtcbiAgICAgICAgICAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL2lucHV0L18xMycsXG4gICAgICAgICAgJ0Z1bmMvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGxfMS9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9pbnB1dC9fMTQnLFxuICAgICAgICAgICdGdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsXzEvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvaW5wdXQvXzE1J1xuICAgICAgICBdLFxuICAgICAgICBhdHRyOiB7XG4gICAgICAgICAgVG91dDoge3R5cGU6ICdEVF9JTlQzMid9LFxuICAgICAgICAgIFRpbjoge3R5cGU6ICdEVF9TVFJJTkcnfSxcbiAgICAgICAgICBfaGFzX21hbnVhbF9jb250cm9sX2RlcGVuZGVuY2llczoge2I6IHRydWV9XG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6XG4gICAgICAgICAgICAnRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbF8xL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsL291dHB1dF9jb250cm9sX25vZGUvXzE3JyxcbiAgICAgICAgb3A6ICdOb09wJyxcbiAgICAgICAgaW5wdXQ6IFtcbiAgICAgICAgICAnXlN0YXRlZnVsUGFydGl0aW9uZWRDYWxsXzEvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwva2V5X3ZhbHVlX2luaXQ5NC9Mb29rdXBUYWJsZUltcG9ydFYyJ1xuICAgICAgICBdLFxuICAgICAgICBhdHRyOiB7X2hhc19tYW51YWxfY29udHJvbF9kZXBlbmRlbmNpZXM6IHtiOiB0cnVlfX1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgIG5hbWU6ICdGdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsXzEvb3V0cHV0X2NvbnRyb2xfbm9kZS9fOCcsXG4gICAgICAgIG9wOiAnTm9PcCcsXG4gICAgICAgIGlucHV0OiBbXG4gICAgICAgICAgJ15GdW5jL1N0YXRlZnVsUGFydGl0aW9uZWRDYWxsXzEvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvb3V0cHV0X2NvbnRyb2xfbm9kZS9fMTcnXG4gICAgICAgIF0sXG4gICAgICAgIGF0dHI6IHtfaGFzX21hbnVhbF9jb250cm9sX2RlcGVuZGVuY2llczoge2I6IHRydWV9fVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTogJ05vT3AnLFxuICAgICAgICBvcDogJ05vT3AnLFxuICAgICAgICBpbnB1dDogW1xuICAgICAgICAgICdeRnVuYy9TdGF0ZWZ1bFBhcnRpdGlvbmVkQ2FsbC9vdXRwdXRfY29udHJvbF9ub2RlL18yJyxcbiAgICAgICAgICAnXkZ1bmMvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGxfMS9vdXRwdXRfY29udHJvbF9ub2RlL184J1xuICAgICAgICBdLFxuICAgICAgICBhdHRyOiB7XG4gICAgICAgICAgX2hhc19tYW51YWxfY29udHJvbF9kZXBlbmRlbmNpZXM6IHtiOiB0cnVlfSxcbiAgICAgICAgICBfYWNkX2Z1bmN0aW9uX2NvbnRyb2xfb3V0cHV0OiB7YjogdHJ1ZX1cbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgbmFtZTogJ0lkZW50aXR5JyxcbiAgICAgICAgb3A6ICdJZGVudGl0eScsXG4gICAgICAgIGlucHV0OiBbXG4gICAgICAgICAgJ0Z1bmMvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGwvb3V0cHV0L18xJyxcbiAgICAgICAgICAnXkZ1bmMvU3RhdGVmdWxQYXJ0aXRpb25lZENhbGxfMS9vdXRwdXRfY29udHJvbF9ub2RlL184JywgJ15Ob09wJ1xuICAgICAgICBdLFxuICAgICAgICBhdHRyOiB7VDoge3R5cGU6ICdEVF9SRVNPVVJDRSd9fVxuICAgICAgfVxuICAgIF0sXG4gICAgdmVyc2lvbnM6IHtwcm9kdWNlcjogMTI0MH1cbiAgfSxcbiAgaW5pdGlhbGl6ZXJTaWduYXR1cmU6IHtcbiAgICBvdXRwdXRzOiB7XG4gICAgICAnSWRlbnRpdHk6MCc6IHtcbiAgICAgICAgbmFtZTogJ0lkZW50aXR5OjAnLFxuICAgICAgICBkdHlwZTogJ0RUX1JFU09VUkNFJyxcbiAgICAgICAgdGVuc29yU2hhcGU6IHt9LFxuICAgICAgICByZXNvdXJjZUlkOiA2NlxuICAgICAgfVxuICAgIH1cbiAgfVxufTtcbiJdfQ==