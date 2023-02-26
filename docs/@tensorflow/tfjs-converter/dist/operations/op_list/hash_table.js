/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
export const json = [
    {
        'tfOpName': 'HashTable',
        'category': 'hash_table',
        'inputs': [],
        'attrs': [
            {
                'tfName': 'shared_name',
                'name': 'sharedName',
                'type': 'string'
            },
            {
                'tfName': 'use_node_name_sharing',
                'name': 'useNodeNameSharing',
                'type': 'bool'
            },
            {
                'tfName': 'key_dtype',
                'name': 'keyDType',
                'type': 'dtype'
            },
            {
                'tfName': 'value_dtype',
                'name': 'valueDType',
                'type': 'dtype'
            }
        ]
    },
    {
        'tfOpName': 'HashTableV2',
        'category': 'hash_table',
        'inputs': [],
        'attrs': [
            {
                'tfName': 'shared_name',
                'name': 'sharedName',
                'type': 'string'
            },
            {
                'tfName': 'use_node_name_sharing',
                'name': 'useNodeNameSharing',
                'type': 'bool'
            },
            {
                'tfName': 'key_dtype',
                'name': 'keyDType',
                'type': 'dtype'
            },
            {
                'tfName': 'value_dtype',
                'name': 'valueDType',
                'type': 'dtype'
            }
        ]
    },
    {
        'tfOpName': 'LookupTableImport',
        'category': 'hash_table',
        'inputs': [
            {
                'start': 0,
                'name': 'tableHandle',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'keys',
                'type': 'tensor'
            },
            {
                'start': 2,
                'name': 'values',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'Tin',
                'name': 'tIn',
                'type': 'dtype',
                'notSupported': true
            },
            {
                'tfName': 'Tout',
                'name': 'tOut',
                'type': 'dtype',
                'notSupported': true
            }
        ]
    },
    {
        'tfOpName': 'LookupTableImportV2',
        'category': 'hash_table',
        'inputs': [
            {
                'start': 0,
                'name': 'tableHandle',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'keys',
                'type': 'tensor'
            },
            {
                'start': 2,
                'name': 'values',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'Tin',
                'name': 'tIn',
                'type': 'dtype',
                'notSupported': true
            },
            {
                'tfName': 'Tout',
                'name': 'tOut',
                'type': 'dtype',
                'notSupported': true
            }
        ]
    },
    {
        'tfOpName': 'LookupTableFind',
        'category': 'hash_table',
        'inputs': [
            {
                'start': 0,
                'name': 'tableHandle',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'keys',
                'type': 'tensor'
            },
            {
                'start': 2,
                'name': 'defaultValue',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'Tin',
                'name': 'tIn',
                'type': 'dtype',
                'notSupported': true
            },
            {
                'tfName': 'Tout',
                'name': 'tOut',
                'type': 'dtype',
                'notSupported': true
            }
        ]
    },
    {
        'tfOpName': 'LookupTableFindV2',
        'category': 'hash_table',
        'inputs': [
            {
                'start': 0,
                'name': 'tableHandle',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'keys',
                'type': 'tensor'
            },
            {
                'start': 2,
                'name': 'defaultValue',
                'type': 'tensor'
            }
        ],
        'attrs': [
            {
                'tfName': 'Tin',
                'name': 'tIn',
                'type': 'dtype',
                'notSupported': true
            },
            {
                'tfName': 'Tout',
                'name': 'tOut',
                'type': 'dtype',
                'notSupported': true
            }
        ]
    },
    {
        'tfOpName': 'LookupTableSize',
        'category': 'hash_table',
        'inputs': [
            {
                'start': 0,
                'name': 'tableHandle',
                'type': 'tensor'
            }
        ]
    },
    {
        'tfOpName': 'LookupTableSizeV2',
        'category': 'hash_table',
        'inputs': [
            {
                'start': 0,
                'name': 'tableHandle',
                'type': 'tensor'
            }
        ]
    },
    {
        'tfOpName': 'InitializeTable',
        'category': 'hash_table',
        'inputs': [
            {
                'start': 0,
                'name': 'tableHandle',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'keys',
                'type': 'tensor'
            },
            {
                'start': 2,
                'name': 'values',
                'type': 'tensor'
            }
        ]
    },
    {
        'tfOpName': 'InitializeTableV2',
        'category': 'hash_table',
        'inputs': [
            {
                'start': 0,
                'name': 'tableHandle',
                'type': 'tensor'
            },
            {
                'start': 1,
                'name': 'keys',
                'type': 'tensor'
            },
            {
                'start': 2,
                'name': 'values',
                'type': 'tensor'
            }
        ]
    }
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaGFzaF90YWJsZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29udmVydGVyL3NyYy9vcGVyYXRpb25zL29wX2xpc3QvaGFzaF90YWJsZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFDQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFJSCxNQUFNLENBQUMsTUFBTSxJQUFJLEdBQWU7SUFDOUI7UUFDRSxVQUFVLEVBQUUsV0FBVztRQUN2QixVQUFVLEVBQUUsWUFBWTtRQUN4QixRQUFRLEVBQUUsRUFBRTtRQUNaLE9BQU8sRUFBRTtZQUNQO2dCQUNFLFFBQVEsRUFBRSxhQUFhO2dCQUN2QixNQUFNLEVBQUUsWUFBWTtnQkFDcEIsTUFBTSxFQUFFLFFBQVE7YUFDakI7WUFDRDtnQkFDRSxRQUFRLEVBQUUsdUJBQXVCO2dCQUNqQyxNQUFNLEVBQUUsb0JBQW9CO2dCQUM1QixNQUFNLEVBQUUsTUFBTTthQUNmO1lBQ0Q7Z0JBQ0UsUUFBUSxFQUFFLFdBQVc7Z0JBQ3JCLE1BQU0sRUFBRSxVQUFVO2dCQUNsQixNQUFNLEVBQUUsT0FBTzthQUNoQjtZQUNEO2dCQUNFLFFBQVEsRUFBRSxhQUFhO2dCQUN2QixNQUFNLEVBQUUsWUFBWTtnQkFDcEIsTUFBTSxFQUFFLE9BQU87YUFDaEI7U0FDRjtLQUNGO0lBQ0Q7UUFDRSxVQUFVLEVBQUUsYUFBYTtRQUN6QixVQUFVLEVBQUUsWUFBWTtRQUN4QixRQUFRLEVBQUUsRUFBRTtRQUNaLE9BQU8sRUFBRTtZQUNQO2dCQUNFLFFBQVEsRUFBRSxhQUFhO2dCQUN2QixNQUFNLEVBQUUsWUFBWTtnQkFDcEIsTUFBTSxFQUFFLFFBQVE7YUFDakI7WUFDRDtnQkFDRSxRQUFRLEVBQUUsdUJBQXVCO2dCQUNqQyxNQUFNLEVBQUUsb0JBQW9CO2dCQUM1QixNQUFNLEVBQUUsTUFBTTthQUNmO1lBQ0Q7Z0JBQ0UsUUFBUSxFQUFFLFdBQVc7Z0JBQ3JCLE1BQU0sRUFBRSxVQUFV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QUFFLFFBQVE7Z0JBQ2hCLE1BQU0sRUFBRSxRQUFRO2FBQ2pCO1NBQ0Y7S0FDRjtJQUNEO1FBQ0UsVUFBVSxFQUFFLG1CQUFtQjtRQUMvQixVQUFVLEVBQUUsWUFBWTtRQUN4QixRQUFRLEVBQUU7WUFDUjtnQkFDRSxPQUFPLEVBQUUsQ0FBQztnQkFDVixNQUFNLEVBQUUsYUFBYTtnQkFDckIsTUFBTSxFQUFFLFFBQVE7YUFDakI7WUFDRDtnQkFDRSxPQUFPLEVBQUUsQ0FBQztnQkFDVixNQUFNLEVBQUUsTUFBTTtnQkFDZCxNQUFNLEVBQUUsUUFBUTthQUNqQjtZQUNEO2dCQUNFLE9BQU8sRUFBRSxDQUFDO2dCQUNWLE1BQU0sRUFBRSxRQUFRO2dCQUNoQixNQUFNLEVBQUUsUUFBUTthQUNqQjtTQUNGO0tBQ0Y7Q0FDRixDQUNBIiwic291cmNlc0NvbnRlbnQiOlsiXG4vKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMyBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7T3BNYXBwZXJ9IGZyb20gJy4uL3R5cGVzJztcblxuZXhwb3J0IGNvbnN0IGpzb246IE9wTWFwcGVyW10gPSBbXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnSGFzaFRhYmxlJyxcbiAgICAnY2F0ZWdvcnknOiAnaGFzaF90YWJsZScsXG4gICAgJ2lucHV0cyc6IFtdLFxuICAgICdhdHRycyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdzaGFyZWRfbmFtZScsXG4gICAgICAgICduYW1lJzogJ3NoYXJlZE5hbWUnLFxuICAgICAgICAndHlwZSc6ICdzdHJpbmcnXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3VzZV9ub2RlX25hbWVfc2hhcmluZycsXG4gICAgICAgICduYW1lJzogJ3VzZU5vZGVOYW1lU2hhcmluZycsXG4gICAgICAgICd0eXBlJzogJ2Jvb2wnXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ2tleV9kdHlwZScsXG4gICAgICAgICduYW1lJzogJ2tleURUeXBlJyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ3ZhbHVlX2R0eXBlJyxcbiAgICAgICAgJ25hbWUnOiAndmFsdWVEVHlwZScsXG4gICAgICAgICd0eXBlJzogJ2R0eXBlJ1xuICAgICAgfVxuICAgIF1cbiAgfSxcbiAge1xuICAgICd0Zk9wTmFtZSc6ICdIYXNoVGFibGVWMicsXG4gICAgJ2NhdGVnb3J5JzogJ2hhc2hfdGFibGUnLFxuICAgICdpbnB1dHMnOiBbXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnc2hhcmVkX25hbWUnLFxuICAgICAgICAnbmFtZSc6ICdzaGFyZWROYW1lJyxcbiAgICAgICAgJ3R5cGUnOiAnc3RyaW5nJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICd1c2Vfbm9kZV9uYW1lX3NoYXJpbmcnLFxuICAgICAgICAnbmFtZSc6ICd1c2VOb2RlTmFtZVNoYXJpbmcnLFxuICAgICAgICAndHlwZSc6ICdib29sJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdrZXlfZHR5cGUnLFxuICAgICAgICAnbmFtZSc6ICdrZXlEVHlwZScsXG4gICAgICAgICd0eXBlJzogJ2R0eXBlJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICd2YWx1ZV9kdHlwZScsXG4gICAgICAgICduYW1lJzogJ3ZhbHVlRFR5cGUnLFxuICAgICAgICAndHlwZSc6ICdkdHlwZSdcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnTG9va3VwVGFibGVJbXBvcnQnLFxuICAgICdjYXRlZ29yeSc6ICdoYXNoX3RhYmxlJyxcbiAgICAnaW5wdXRzJzogW1xuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAwLFxuICAgICAgICAnbmFtZSc6ICd0YWJsZUhhbmRsZScsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDEsXG4gICAgICAgICduYW1lJzogJ2tleXMnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAyLFxuICAgICAgICAnbmFtZSc6ICd2YWx1ZXMnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnVGluJyxcbiAgICAgICAgJ25hbWUnOiAndEluJyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdUb3V0JyxcbiAgICAgICAgJ25hbWUnOiAndE91dCcsXG4gICAgICAgICd0eXBlJzogJ2R0eXBlJyxcbiAgICAgICAgJ25vdFN1cHBvcnRlZCc6IHRydWVcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnTG9va3VwVGFibGVJbXBvcnRWMicsXG4gICAgJ2NhdGVnb3J5JzogJ2hhc2hfdGFibGUnLFxuICAgICdpbnB1dHMnOiBbXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDAsXG4gICAgICAgICduYW1lJzogJ3RhYmxlSGFuZGxlJyxcbiAgICAgICAgJ3R5cGUnOiAndGVuc29yJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMSxcbiAgICAgICAgJ25hbWUnOiAna2V5cycsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDIsXG4gICAgICAgICduYW1lJzogJ3ZhbHVlcycsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH1cbiAgICBdLFxuICAgICdhdHRycyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdUaW4nLFxuICAgICAgICAnbmFtZSc6ICd0SW4nLFxuICAgICAgICAndHlwZSc6ICdkdHlwZScsXG4gICAgICAgICdub3RTdXBwb3J0ZWQnOiB0cnVlXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAndGZOYW1lJzogJ1RvdXQnLFxuICAgICAgICAnbmFtZSc6ICd0T3V0JyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfVxuICAgIF1cbiAgfSxcbiAge1xuICAgICd0Zk9wTmFtZSc6ICdMb29rdXBUYWJsZUZpbmQnLFxuICAgICdjYXRlZ29yeSc6ICdoYXNoX3RhYmxlJyxcbiAgICAnaW5wdXRzJzogW1xuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAwLFxuICAgICAgICAnbmFtZSc6ICd0YWJsZUhhbmRsZScsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDEsXG4gICAgICAgICduYW1lJzogJ2tleXMnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAyLFxuICAgICAgICAnbmFtZSc6ICdkZWZhdWx0VmFsdWUnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnVGluJyxcbiAgICAgICAgJ25hbWUnOiAndEluJyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdUb3V0JyxcbiAgICAgICAgJ25hbWUnOiAndE91dCcsXG4gICAgICAgICd0eXBlJzogJ2R0eXBlJyxcbiAgICAgICAgJ25vdFN1cHBvcnRlZCc6IHRydWVcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnTG9va3VwVGFibGVGaW5kVjInLFxuICAgICdjYXRlZ29yeSc6ICdoYXNoX3RhYmxlJyxcbiAgICAnaW5wdXRzJzogW1xuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAwLFxuICAgICAgICAnbmFtZSc6ICd0YWJsZUhhbmRsZScsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDEsXG4gICAgICAgICduYW1lJzogJ2tleXMnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAyLFxuICAgICAgICAnbmFtZSc6ICdkZWZhdWx0VmFsdWUnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXSxcbiAgICAnYXR0cnMnOiBbXG4gICAgICB7XG4gICAgICAgICd0Zk5hbWUnOiAnVGluJyxcbiAgICAgICAgJ25hbWUnOiAndEluJyxcbiAgICAgICAgJ3R5cGUnOiAnZHR5cGUnLFxuICAgICAgICAnbm90U3VwcG9ydGVkJzogdHJ1ZVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3RmTmFtZSc6ICdUb3V0JyxcbiAgICAgICAgJ25hbWUnOiAndE91dCcsXG4gICAgICAgICd0eXBlJzogJ2R0eXBlJyxcbiAgICAgICAgJ25vdFN1cHBvcnRlZCc6IHRydWVcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnTG9va3VwVGFibGVTaXplJyxcbiAgICAnY2F0ZWdvcnknOiAnaGFzaF90YWJsZScsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAndGFibGVIYW5kbGUnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXVxuICB9LFxuICB7XG4gICAgJ3RmT3BOYW1lJzogJ0xvb2t1cFRhYmxlU2l6ZVYyJyxcbiAgICAnY2F0ZWdvcnknOiAnaGFzaF90YWJsZScsXG4gICAgJ2lucHV0cyc6IFtcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMCxcbiAgICAgICAgJ25hbWUnOiAndGFibGVIYW5kbGUnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXVxuICB9LFxuICB7XG4gICAgJ3RmT3BOYW1lJzogJ0luaXRpYWxpemVUYWJsZScsXG4gICAgJ2NhdGVnb3J5JzogJ2hhc2hfdGFibGUnLFxuICAgICdpbnB1dHMnOiBbXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDAsXG4gICAgICAgICduYW1lJzogJ3RhYmxlSGFuZGxlJyxcbiAgICAgICAgJ3R5cGUnOiAndGVuc29yJ1xuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3N0YXJ0JzogMSxcbiAgICAgICAgJ25hbWUnOiAna2V5cycsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDIsXG4gICAgICAgICduYW1lJzogJ3ZhbHVlcycsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH1cbiAgICBdXG4gIH0sXG4gIHtcbiAgICAndGZPcE5hbWUnOiAnSW5pdGlhbGl6ZVRhYmxlVjInLFxuICAgICdjYXRlZ29yeSc6ICdoYXNoX3RhYmxlJyxcbiAgICAnaW5wdXRzJzogW1xuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAwLFxuICAgICAgICAnbmFtZSc6ICd0YWJsZUhhbmRsZScsXG4gICAgICAgICd0eXBlJzogJ3RlbnNvcidcbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdzdGFydCc6IDEsXG4gICAgICAgICduYW1lJzogJ2tleXMnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAnc3RhcnQnOiAyLFxuICAgICAgICAnbmFtZSc6ICd2YWx1ZXMnLFxuICAgICAgICAndHlwZSc6ICd0ZW5zb3InXG4gICAgICB9XG4gICAgXVxuICB9XG5dXG47XG4iXX0=