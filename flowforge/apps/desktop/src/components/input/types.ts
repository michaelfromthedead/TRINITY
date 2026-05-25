export interface SelectOption {
  name: string
  value: string
}

export interface MultiSelectOption extends SelectOption {
  disabled?: boolean
}
