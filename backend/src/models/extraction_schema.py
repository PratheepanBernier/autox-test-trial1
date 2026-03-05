from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Location(BaseModel):
    name: Optional[str] = Field(None, description="Location name")
    address: Optional[str] = Field(None, description="Full address")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State")
    zip_code: Optional[str] = Field(None, description="ZIP code")
    country: Optional[str] = Field(None, description="Country")
    appointment_time: Optional[str] = Field(None, description="Appointment time")
    po_number: Optional[str] = Field(None, description="PO/Container number")

class CommodityItem(BaseModel):
    commodity_name: Optional[str] = Field(None, description="Name of the commodity")
    weight: Optional[str] = Field(None, description="Weight with unit (e.g., '56000.00 lbs')")
    quantity: Optional[str] = Field(None, description="Quantity with unit (e.g., '10000 units')")
    description: Optional[str] = Field(None, description="Description of the commodity")

class CarrierInfo(BaseModel):
    carrier_name: Optional[str] = Field(None, description="Name of the carrier company")
    mc_number: Optional[str] = Field(None, description="MC number")
    phone: Optional[str] = Field(None, description="Carrier phone number")
    email: Optional[str] = Field(None, description="Carrier email")

class DriverInfo(BaseModel):
    driver_name: Optional[str] = Field(None, description="Driver's name")
    cell_number: Optional[str] = Field(None, description="Driver's cell phone")
    truck_number: Optional[str] = Field(None, description="Truck number")
    trailer_number: Optional[str] = Field(None, description="Trailer number")

class RateInfo(BaseModel):
    total_rate: Optional[float] = Field(None, description="Total rate amount")
    currency: Optional[str] = Field(None, description="Currency code (e.g., USD)")
    rate_breakdown: Optional[Dict[str, Any]] = Field(None, description="Breakdown of rates by type")

class ShipmentData(BaseModel):
    # Core IDs
    reference_id: Optional[str] = Field(None, description="Reference ID or Load ID")
    load_id: Optional[str] = Field(None, description="Load ID")
    po_number: Optional[str] = Field(None, description="PO number")
    
    # Parties
    shipper: Optional[str] = Field(None, description="Shipper name or customer name")
    consignee: Optional[str] = Field(None, description="Consignee name")
    carrier: Optional[CarrierInfo] = Field(None, description="Carrier information")
    driver: Optional[DriverInfo] = Field(None, description="Driver information")
    
    # Locations
    pickup: Optional[Location] = Field(None, description="Pickup location details")
    drop: Optional[Location] = Field(None, description="Drop/delivery location details")
    
    # Dates
    shipping_date: Optional[str] = Field(None, description="Shipping date")
    delivery_date: Optional[str] = Field(None, description="Delivery date")
    created_on: Optional[str] = Field(None, description="Document creation date")
    booking_date: Optional[str] = Field(None, description="Booking date")
    
    # Equipment
    equipment_type: Optional[str] = Field(None, description="Equipment type (e.g., Flatbed, Van, Reefer)")
    equipment_size: Optional[str] = Field(None, description="Equipment size in feet")
    load_type: Optional[str] = Field(None, description="Load type (e.g., FTL, LTL)")
    
    # Commodities
    commodities: Optional[List[CommodityItem]] = Field(None, description="List of commodities being shipped")
    
    # Rates
    rate_info: Optional[RateInfo] = Field(None, description="Rate information")
    
    # Instructions
    special_instructions: Optional[str] = Field(None, description="Special instructions")
    shipper_instructions: Optional[str] = Field(None, description="Shipper-specific instructions")
    carrier_instructions: Optional[str] = Field(None, description="Carrier-specific instructions")
    
    # Dispatcher/Contact
    dispatcher_name: Optional[str] = Field(None, description="Dispatcher name")
    dispatcher_phone: Optional[str] = Field(None, description="Dispatcher phone")
    dispatcher_email: Optional[str] = Field(None, description="Dispatcher email")
    
    # Other fields (for unknown/new fields)
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional data not covered by other fields")

class ExtractionResponse(BaseModel):
    data: ShipmentData
    document_id: Optional[str] = None
